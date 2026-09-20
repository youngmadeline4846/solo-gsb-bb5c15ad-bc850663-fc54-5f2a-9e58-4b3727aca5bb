"""Mixed-integer blend optimizer.

Model
-----
Variables z_i >= 0: units drawn from source i.  When source i declares a
``batch_size`` b_i the variable is constrained integer and the volume is
``x_i = b_i * z_i``; otherwise z_i is continuous and ``x_i = z_i``.

A source that declares ``minimum_run_volume`` m_i additionally gets a binary
enable variable y_i in {0, 1} with the big-M constraints

    x_i <= U_i * y_i            (nothing may be drawn while disabled)
    x_i >= m_i * y_i            (once enabled, at least m_i must be drawn)

so its usage can only be 0 or >= m_i; U_i is the largest drawable volume
(the capped ``max_volume`` floored to the batch lattice).  When the request
sets ``minimum_active_sources`` K, the enable variables of all threshold
sources are coupled by

    sum(y_i) >= K.

A source that declares ``price_tiers`` instead of a flat ``cost`` is priced
on ordered marginal blocks.  For source i with tiers t = 0, ..., p_i - 1,
continuous tier-usage variables q_{i,t} >= 0 and binary tier-saturated
variables w_{i,t} in {0, 1} are added, all in the SAME MIP:

    x_i              = sum_t q_{i,t}
    0 <= q_{i,t}    <= cap_{i,t}
    q_{i,t}         >= cap_{i,t} * w_{i,t}      (w_t = 1 means block t full)
    q_{i,t+1}       <= cap_{i,t+1} * w_{i,t}    (volume reaches block t+1
                                                 only once block t is full)

so the blocks are consumed strictly in declaration order: every unit in a
later block is matched by a full preceding block, and a later block opens
only after the one before it is exhausted.  The currently open block may be
partially filled (its w stays 0).  ``cap`` is the declared tier capacity,
except that a ``null`` capacity on the last tier spans the remaining supply
up to ``max_volume``.  The objective contribution is

    sum_t price_{i,t} * q_{i,t},

which replaces ``cost_i * x_i``.  Batched sources use the same block
variables: batch water spanning a tier boundary is split marginally between
the two blocks.

    minimize    sum(cost_i * x_i)   or   sum(price_{i,t} * q_{i,t})
    subject to  sum(x_i)                       = target_volume
                lo_j * target_volume <= sum(conc_ij * x_i) <= hi_j * target_volume
                x_i                            <= max_volume_i
                z_i in Z>=0 for batched sources, z_i in R>=0 otherwise
                x_i in {0} or [m_i, U_i] for threshold sources (via y_i)
                sum(y_i) >= K when minimum_active_sources is set
                tier usage ordered by binary w variables (tiered sources)

Continuous, batched, commissioned and tiered sources are solved together in
a single mixed-integer program; a continuous optimum is never rounded or
trimmed afterwards (small usages are never pruned in a post-processing pass).

When several optima share the same minimal cost, the usage vector ordered by
ascending source id is minimized lexicographically via a sequence of MIPs.

If the problem is infeasible, the achievable (min, max) concentration range of
every indicator is reported, projected onto the *same* mixed-integer feasible
set (supply caps, volume, batch, commissioning and tier-ordering constraints).

Supply-reduction contingency plans (``solve_contingency``) use the same
mixed-integer feasible set with tightened supply caps (which clamp every
tier capacity), but optimize a lexicographic objective: first minimize the
total absolute volume change versus the stored allocation, then minimize
cost, and finally the source-id-ordered usage vector lexicographically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

INDICATORS: Tuple[str, ...] = ("calcium", "magnesium", "alkalinity")

# Half-width (relative) of the interval used to pin a variable after its
# lexicographic minimum has been found.  Comfortably above solver noise and
# below the 6-decimal output rounding for typical magnitudes.
_FIX_RTOL = 1e-8
# Fallback cost slack (relative) if the exact cost-equality refinement fails.
_COST_RTOL = 1e-9


@dataclass
class SourceData:
    id: str
    max_volume: float
    calcium: float
    magnesium: float
    alkalinity: float
    cost: Optional[float] = None
    batch_size: Optional[float] = None
    minimum_run_volume: Optional[float] = None
    # Ordered marginal price blocks (capacity, unit price); the last
    # capacity may be None to span the remaining supply up to max_volume.
    # When non-empty, cost must be None and the block prices replace it.
    price_tiers: Optional[List[Tuple[Optional[float], float]]] = None


@dataclass
class _Model:
    """Constraint matrix description in the (possibly scaled) variable space.

    Variable layout: ``(z_0..z_{n-1}, y_0..y_{m-1}, w/q of tiered sources
    in id-sorted source order, ...)``.  Volumes are recovered as
    ``x = scales * z`` for the first ``n`` columns; ``integrality`` marks
    the batch-count variables, the binary enable variables and the binary
    tier-ordering variables.

    ``A_struct``/``b_struct`` hold the commissioning (minimum-run and
    active-count) rows; ``A_struct_eq``/``b_struct_eq`` hold the structural
    equalities (one x_i = sum q per tiered source).  Both groups are part of
    *every* projection of the feasible set, including the indicator-free
    infeasibility analysis.  ``A_ub``/``b_ub`` hold the indicator windows,
    which that analysis deliberately drops.
    """

    n: int
    n_vars: int
    scales: np.ndarray           # batch_size per source, 1.0 when continuous
    integrality: np.ndarray      # 1 for integer/binary variables, else 0
    bounds: List[Tuple[float, float]]
    A_eq: np.ndarray
    b_eq: np.ndarray
    A_ub: Optional[np.ndarray]
    b_ub: Optional[np.ndarray]
    A_struct: Optional[np.ndarray]
    b_struct: Optional[np.ndarray]
    A_struct_eq: Optional[np.ndarray]
    b_struct_eq: Optional[np.ndarray]
    y_col: Dict[int, int]        # source index -> enable-variable column
    threshold: np.ndarray        # bool per source: declares minimum_run_volume
    minimum_active: int          # required active threshold sources (>= 0)
    minimum_run: np.ndarray      # minimum-run volume per source (0 if unset)
    # Tier columns per source: list aligned with source indices; each entry
    # is (w_col_start, q_col_start, p_i) or None for untiered sources.
    tier_cols: List[Optional[Tuple[int, int, int]]] = field(default_factory=list)
    # Effective tier capacities (null last tier expanded to max_volume),
    # aligned with tier columns; entries are None for untiered sources.
    tier_caps: List[Optional[np.ndarray]] = field(default_factory=list)
    tier_prices: List[Optional[np.ndarray]] = field(default_factory=list)


@dataclass
class SolveOutput:
    status: str  # "optimal" | "infeasible"
    volumes: Optional[np.ndarray] = None       # aligned with the sorted sources
    total_cost: Optional[float] = None
    indicators: Optional[Dict[str, float]] = None
    achievable_ranges: Optional[Dict[str, dict]] = None
    message: Optional[str] = None


@dataclass
class ContingencyOutput:
    """Result of a supply-reduction contingency re-optimization."""

    status: str  # "optimal" | "infeasible"
    volumes: Optional[np.ndarray] = None       # aligned with the sorted sources
    deltas: Optional[np.ndarray] = None        # new volume - original volume
    total_adjustment: Optional[float] = None   # sum of absolute deltas
    total_cost: Optional[float] = None
    cost_delta: Optional[float] = None         # new cost - original cost
    indicators: Optional[Dict[str, float]] = None
    achievable_ranges: Optional[Dict[str, dict]] = None
    message: Optional[str] = None


# --------------------------------------------------------------------------- #
# tier helpers
# --------------------------------------------------------------------------- #

def _tier_clamps(source: SourceData,
                 cap_volume: Optional[float] = None) -> np.ndarray:
    """Effective per-tier capacities in m3 for a tiered source.

    The declared ``null`` last tier spans the supply left after the finite
    tiers under the applicable cap (``max_volume`` normally, the reduced
    contingency cap otherwise), clamped at zero when the finite tiers alone
    already reach that cap.
    """
    assert source.price_tiers is not None
    supply = float(source.max_volume if cap_volume is None else cap_volume)
    declared = list(source.price_tiers)
    caps = np.array(
        [supply if (t == len(declared) - 1 and c is None) else float(c)
         for t, (c, _p) in enumerate(declared)],
        dtype=float,
    )
    # A null last tier covers only the supply beyond the preceding blocks.
    if declared[-1][0] is None and len(caps) > 1:
        caps[-1] = max(0.0, supply - float(caps[:-1].sum()))
    return caps


def tier_breakdown(
    source: SourceData, volume: float, cap_volume: Optional[float] = None
) -> List[Tuple[float, float, float]]:
    """Split a source usage across its declared marginal tiers.

    Returns ``(used_volume, unit_price, cost)`` per tier in declaration
    order.  Tiers are filled sequentially; for a tiered source the split is
    uniquely determined by the total volume regardless of prices.
    """
    tiers = source.price_tiers or []
    caps = _tier_clamps(source, cap_volume) if tiers else np.array([])
    lines: List[Tuple[float, float, float]] = []
    remaining = max(0.0, float(volume))
    for (declared_cap, price), eff_cap in zip(tiers, caps):
        used = min(remaining, float(eff_cap)) if remaining > 0.0 else 0.0
        remaining = max(0.0, remaining - used)
        lines.append((used, float(price), used * float(price)))
    return lines


def source_cost(
    source: SourceData, volume: float, cap_volume: Optional[float] = None
) -> float:
    """Flat or tiered cost of drawing ``volume`` from one source."""
    if source.price_tiers:
        return float(sum(c for _u, _p, c in tier_breakdown(source, volume, cap_volume)))
    return float(volume * (source.cost or 0.0))


# --------------------------------------------------------------------------- #
# model construction
# --------------------------------------------------------------------------- #

def _scales(sources: Sequence[SourceData]) -> np.ndarray:
    return np.array(
        [1.0 if s.batch_size is None else float(s.batch_size) for s in sources],
        dtype=float,
    )


def _integrality(sources: Sequence[SourceData]) -> np.ndarray:
    # Decided from the flag, not from scale == 1 (batch_size 1.0 stays integer).
    return np.array([0.0 if s.batch_size is None else 1.0 for s in sources],
                    dtype=float)


def _threshold_mask(sources: Sequence[SourceData]) -> np.ndarray:
    return np.array([s.minimum_run_volume is not None for s in sources], dtype=bool)


def _minimum_run(sources: Sequence[SourceData]) -> np.ndarray:
    return np.array(
        [0.0 if s.minimum_run_volume is None else float(s.minimum_run_volume)
         for s in sources],
        dtype=float,
    )


def _source_bounds(sources: Sequence[SourceData], scales: np.ndarray,
                   caps: Optional[Mapping[str, float]] = None
                   ) -> List[Tuple[float, float]]:
    bounds: List[Tuple[float, float]] = []
    for s, scale in zip(sources, scales):
        cap = float(s.max_volume) if caps is None else float(caps.get(s.id, s.max_volume))
        if s.batch_size is None:
            hi = cap
        else:
            hi = float(math.floor(cap / scale + 1e-12))
        bounds.append((0.0, hi))
    return bounds


def _indicator_constraints(
    sources: Sequence[SourceData],
    target_volume: float,
    limits: Mapping[str, Tuple[float, float]],
    scales: np.ndarray,
    n_vars: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Indicator-window rows in z-space, zero-padded over the extra columns:
    ``lo_j*V <= sum(conc_ij * scale_i * z_i) <= hi_j*V``."""
    n = len(sources)
    rows, rhs = [], []
    for name in INDICATORS:
        rng = limits.get(name)
        if rng is None:
            continue
        lo, hi = rng
        coeff = np.zeros(n_vars)
        coeff[:n] = (
            np.array([getattr(s, name) for s in sources], dtype=float) * scales
        )
        rows.append(-coeff)
        rhs.append(-lo * target_volume)
        rows.append(coeff)
        rhs.append(hi * target_volume)
    if not rows:
        return None, None
    return np.asarray(rows), np.asarray(rhs)


def _commissioning_constraints(
    sources: Sequence[SourceData],
    scales: np.ndarray,
    bounds: List[Tuple[float, float]],
    threshold: np.ndarray,
    minimum_run: np.ndarray,
    minimum_active: int,
    n_core: int,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]],
           np.ndarray, Dict[int, int], np.ndarray]:
    """Build the minimum-run big-M rows and the active-count row.

    Returns (A_struct, b_struct, extra_bounds, y_col, extra_integrality,
    minimum_run).  Rows live in the full variable space of width
    ``n_core``; the enable columns occupy positions n .. n+m-1.
    """
    n = len(sources)
    idx = np.flatnonzero(threshold)
    m = idx.size
    y_col = {int(i): n + k for k, i in enumerate(idx)}
    extra_bounds = [(0.0, 1.0)] * m
    extra_integ = np.ones(m)

    rows: List[np.ndarray] = []
    rhs: List[float] = []
    for i in idx:
        k = y_col[int(i)]
        # Largest drawable volume (z-units), floored to the batch lattice.
        u_z = bounds[int(i)][1]
        u_vol = scales[int(i)] * u_z
        # x_i - U_i*y_i <= 0: disabled (y=0) forces x_i = 0.
        row = np.zeros(n_core)
        row[int(i)] = scales[int(i)]
        row[k] = -u_vol
        rows.append(row)
        rhs.append(0.0)
        # scale_i*z_i - m_i*y_i >= 0, i.e. -scale_i*z_i + m_i*y_i <= 0:
        # enabled (y=1) forces x_i >= m_i.  For batched sources the minimum
        # feasible enabled batch count is the lattice ceiling of m_i.
        if sources[int(i)].batch_size is None:
            m_eff = float(minimum_run[int(i)])
        else:
            min_batches = int(math.ceil(minimum_run[int(i)] / scales[int(i)] - 1e-12))
            m_eff = min_batches * scales[int(i)]
        row = np.zeros(n_core)
        row[int(i)] = -scales[int(i)]
        row[k] = m_eff
        rows.append(row)
        rhs.append(0.0)
        minimum_run[int(i)] = m_eff
    if minimum_active > 0:
        # -sum(y_i) <= -K: at least K threshold sources must be enabled.
        row = np.zeros(n_core)
        for i in idx:
            row[y_col[int(i)]] = -1.0
        rows.append(row)
        rhs.append(float(-minimum_active))
    A_struct = np.vstack(rows) if rows else np.zeros((0, n_core))
    b_struct = np.asarray(rhs) if rows else np.zeros(0)
    return A_struct, b_struct, extra_bounds, y_col, extra_integ, minimum_run


def _tier_layout(sources: Sequence[SourceData]) -> List[int]:
    """Number of (w, q) tier column pairs per source in id-sorted order."""
    return [len(s.price_tiers) if s.price_tiers else 0 for s in sources]


def _tier_constraints(
    sources: Sequence[SourceData],
    scales: np.ndarray,
    bounds: List[Tuple[float, float]],
    caps: Optional[Mapping[str, float]],
    n: int,
    n_y: int,
    n_core: int,
):
    """Build tier-ordering big-M rows, x_i = sum q equalities and the
    per-source tier column bookkeeping."""
    ub_rows: List[np.ndarray] = []
    ub_rhs: List[float] = []
    eq_rows: List[np.ndarray] = []
    eq_rhs: List[float] = []
    extra_bounds: List[Tuple[float, float]] = []
    extra_integ: List[float] = []
    tier_cols: List[Optional[Tuple[int, int, int]]] = []
    tier_caps: List[Optional[np.ndarray]] = []
    tier_prices: List[Optional[np.ndarray]] = []
    cursor = n + n_y
    for i, s in enumerate(sources):
        if not s.price_tiers:
            tier_cols.append(None)
            tier_caps.append(None)
            tier_prices.append(None)
            continue
        p = len(s.price_tiers)
        cap_volume = None if caps is None else float(caps.get(s.id, s.max_volume))
        eff_caps = _tier_clamps(s, cap_volume)
        # The tier blocks must never describe more volume than the lattice
        # cap permits: batched sources floor the reduced cap per batch, and
        # a null final tier keeps whatever supply remains after the floor.
        if s.batch_size is not None and s.price_tiers[-1][0] is None:
            z_hi = bounds[i][1]
            lattice_cap = scales[i] * z_hi
            finite_sum = float(eff_caps[:-1].sum()) if p > 1 else 0.0
            eff_caps = np.array(
                [float(c) for c in eff_caps[:-1]]
                + [max(0.0, lattice_cap - finite_sum)],
                dtype=float,
            )
        # p continuous block usages q_0..q_{p-1} plus p-1 binaries
        # w_0..w_{p-2}; w_t marks block t as saturated (the final block
        # needs no marker since nothing follows it).
        n_w = p - 1
        w0, q0 = cursor, cursor + n_w
        tier_cols.append((w0, q0, p))
        tier_caps.append(eff_caps)
        tier_prices.append(
            np.array([float(price) for _c, price in s.price_tiers], dtype=float)
        )
        # x_i = sum_t q_{i,t}
        row = np.zeros(n_core)
        row[i] = scales[i]
        for t in range(p):
            row[q0 + t] = -1.0
        eq_rows.append(row)
        eq_rhs.append(0.0)
        extra_bounds.extend((0.0, 1.0) for _ in range(n_w))
        extra_integ.extend(1.0 for _ in range(n_w))
        for t in range(p):
            c_t = float(eff_caps[t])
            if t < p - 1:
                # cap_t*w_t - q_t <= 0: turning w_t on fills block t
                # completely; w_t may stay 0 while q_t is partial.
                row = np.zeros(n_core)
                row[w0 + t] = c_t
                row[q0 + t] = -1.0
                ub_rows.append(row)
                ub_rhs.append(0.0)
                # q_{t+1} - cap_{t+1}*w_t <= 0: the next block receives
                # volume only once block t is saturated.
                row = np.zeros(n_core)
                row[q0 + t + 1] = 1.0
                row[w0 + t] = -float(eff_caps[t + 1])
                ub_rows.append(row)
                ub_rhs.append(0.0)
        extra_bounds.extend((0.0, float(c)) for c in eff_caps)
        extra_integ.extend(0.0 for _ in eff_caps)
        cursor += n_w + p
    A_tier = np.vstack(ub_rows) if ub_rows else None
    b_tier = np.asarray(ub_rhs) if ub_rhs else None
    A_tier_eq = np.vstack(eq_rows) if eq_rows else None
    b_tier_eq = np.asarray(eq_rhs) if eq_rows else None
    return (A_tier, b_tier, A_tier_eq, b_tier_eq, extra_bounds,
            np.asarray(extra_integ) if extra_integ else np.zeros(0),
            tier_cols, tier_caps, tier_prices)


def _build_model(
    sources: Sequence[SourceData],
    target_volume: float,
    limits: Mapping[str, Tuple[float, float]],
    caps: Optional[Mapping[str, float]] = None,
    minimum_active_sources: Optional[int] = None,
) -> _Model:
    """Assemble the (mixed-integer) feasible-set description in z-space."""
    n = len(sources)
    scales = _scales(sources)
    bounds = _source_bounds(sources, scales, caps)
    threshold = _threshold_mask(sources)
    minimum_run = _minimum_run(sources)
    minimum_active = int(minimum_active_sources or 0)
    n_y = int(threshold.sum())
    tier_pairs = _tier_layout(sources)
    # Every block adds a continuous q; all but the final block of each
    # source add one saturated-marker binary w: 2*p - 1 extra columns.
    n_tier_vars = int(sum(max(0, 2 * p - 1) for p in tier_pairs))
    n_core = n + n_y + n_tier_vars

    A_struct, b_struct, y_bounds, y_col, y_integ, minimum_run = (
        _commissioning_constraints(
            sources, scales, bounds, threshold, minimum_run,
            minimum_active, n_core,
        )
    )
    (A_tier, b_tier, A_tier_eq, b_tier_eq, tier_extra_bounds, tier_integ,
     tier_cols, tier_caps, tier_prices) = _tier_constraints(
        sources, scales, bounds, caps, n, n_y, n_core,
    )

    all_bounds = list(bounds) + list(y_bounds) + list(tier_extra_bounds)
    integrality = np.concatenate([
        _integrality(sources), np.asarray(y_integ), tier_integ,
    ])

    # Merge commissioning rows with tier-ordering rows in one inequality
    # group; the structural equalities stay separate.
    struct_parts = [A_struct]
    struct_rhs = [b_struct]
    if A_tier is not None:
        struct_parts.append(A_tier)
        struct_rhs.append(b_tier)
    A_struct_all = np.vstack(struct_parts) if struct_parts else None
    b_struct_all = np.concatenate(struct_rhs) if struct_parts else None

    A_eq = np.zeros((1, n_core))
    A_eq[0, :n] = scales
    b_eq = np.array([float(target_volume)])
    A_ub, b_ub = _indicator_constraints(
        sources, target_volume, limits, scales, n_core
    )
    return _Model(
        n=n,
        n_vars=n_core,
        scales=scales,
        integrality=integrality,
        bounds=all_bounds,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        A_struct=A_struct_all if A_struct_all is not None and A_struct_all.shape[0] else None,
        b_struct=b_struct_all if b_struct_all is not None and b_struct_all.size else None,
        A_struct_eq=A_tier_eq,
        b_struct_eq=b_tier_eq,
        y_col=y_col,
        threshold=threshold,
        minimum_active=minimum_active,
        minimum_run=minimum_run,
        tier_cols=tier_cols,
        tier_caps=tier_caps,
        tier_prices=tier_prices,
    )


_UNSET = object()


def _stack_struct(A_ub, b_ub, model: _Model):
    """Append the commissioning/tier-ordering rows to an inequality group."""
    if model.A_struct is None:
        return A_ub, b_ub
    if A_ub is None:
        return model.A_struct, model.b_struct
    return np.vstack([A_ub, model.A_struct]), np.concatenate([b_ub, model.b_struct])


def _stack_struct_eq(A_eq, b_eq, model: _Model):
    """Append the structural equality rows to an equality group."""
    if model.A_struct_eq is None:
        return A_eq, b_eq
    if A_eq is None:
        return model.A_struct_eq, model.b_struct_eq
    return (
        np.vstack([A_eq, model.A_struct_eq]),
        np.concatenate([b_eq, model.b_struct_eq]),
    )


def _lp(c, model: _Model, bounds=_UNSET, A_ub=_UNSET, b_ub=_UNSET,
        A_eq=_UNSET, b_eq=_UNSET, struct: bool = True):
    """linprog wrapper forwarding the model's variable integrality.

    Constraint arguments default to the model's matrices; pass ``None``
    explicitly to drop that constraint group entirely.  The commissioning
    (minimum-run / active-count), tier-ordering rows and the
    ``x_i = sum q`` tier equalities are always included unless
    ``struct=False`` (used only by the structural-feasibility probes)."""
    a_ub = model.A_ub if A_ub is _UNSET else A_ub
    b_ubv = model.b_ub if b_ub is _UNSET else b_ub
    a_eq = model.A_eq if A_eq is _UNSET else A_eq
    b_eqv = model.b_eq if b_eq is _UNSET else b_eq
    if struct:
        a_ub, b_ubv = _stack_struct(a_ub, b_ubv, model)
        a_eq, b_eqv = _stack_struct_eq(a_eq, b_eqv, model)
    return linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ubv,
        A_eq=a_eq,
        b_eq=b_eqv,
        bounds=model.bounds if bounds is _UNSET else bounds,
        integrality=model.integrality,
        method="highs",
    )


def _cost_row(sources: Sequence[SourceData], model: _Model,
              n_vars: Optional[int] = None) -> np.ndarray:
    """Objective coefficients over the full variable space.

    Flat-cost sources price ``scale_i * z_i``; tiered sources price each
    marginal block ``q_{i,t}`` at its own unit price.  ``n_vars`` may exceed
    ``model.n_vars`` for padded (extended-space) models such as the
    contingency L1 program; the trailing deviation columns get zero weight.
    """
    width = model.n_vars if n_vars is None else n_vars
    c = np.zeros(width)
    for i, s in enumerate(sources):
        info = model.tier_cols[i]
        if info is None:
            c[i] = float(s.cost) * model.scales[i]
        else:
            _w0, q0, p = info
            c[q0:q0 + p] = model.tier_prices[i]
    return c


# --------------------------------------------------------------------------- #
# main solve
# --------------------------------------------------------------------------- #

def solve_blend(
    sources: List[SourceData],
    target_volume: float,
    limits: Dict[str, Tuple[float, float]],
    minimum_active_sources: Optional[int] = None,
) -> SolveOutput:
    """Solve the blending MIP.  ``limits`` maps indicator name -> (min, max)."""
    sources = sorted(sources, key=lambda s: s.id)
    model = _build_model(
        sources, target_volume, limits,
        minimum_active_sources=minimum_active_sources,
    )
    c = _cost_row(sources, model, model.n_vars)

    res = _lp(c, model)
    if res.status == 2:  # provably infeasible
        return _infeasible(sources, model, target_volume, limits)
    if res.status != 0:
        raise RuntimeError(f"MIP solver failed (status {res.status}): {res.message}")

    z_all = _lexicographic_min(c, res.x, model)
    z = z_all[: model.n]
    x = model.scales * z
    total_cost = float(_tiered_cost(sources, model, z_all))
    indicators = {
        name: float(np.array([getattr(s, name) for s in sources]) @ x) / target_volume
        for name in INDICATORS
    }
    return SolveOutput(
        status="optimal",
        volumes=x,
        total_cost=total_cost,
        indicators=indicators,
    )


def _tiered_cost(sources: Sequence[SourceData], model: _Model,
                 z_all: np.ndarray) -> float:
    return float(_cost_row(sources, model) @ z_all)


def _lexicographic_min(c, z_star, model: _Model) -> np.ndarray:
    """Among all cost-optimal solutions, minimize (z_1, ..., z_n)
    lexicographically (indices follow the id-sorted source order; enable and
    tier variables follow after).  Scaling is non-negative so this also
    lexicographically minimizes (x_1, ..., x_n)."""
    # Pin the optimal cost as an equality; z_star satisfies it by construction.
    cost_value = float(c @ z_star)
    A_eq2 = np.vstack([model.A_eq, c.reshape(1, -1)])
    b_eq2 = np.append(model.b_eq, cost_value)
    try:
        return _sequential_min(model, model.A_ub, model.b_ub, A_eq2, b_eq2)
    except RuntimeError:
        # Fallback: relax the cost pin to a tight inequality.
        tol = _COST_RTOL * max(1.0, abs(cost_value))
        cost_row = c.reshape(1, -1)
        A_ub2 = cost_row if model.A_ub is None else np.vstack([model.A_ub, cost_row])
        b_ub2 = (
            np.array([cost_value + tol])
            if model.b_ub is None
            else np.append(model.b_ub, cost_value + tol)
        )
        return _sequential_min(model, A_ub2, b_ub2, model.A_eq, model.b_eq)


def _sequential_min(model: _Model, A_ub, b_ub, A_eq, b_eq,
                    bounds=None, n_coords: Optional[int] = None) -> np.ndarray:
    """Lexicographically minimize (z_0, ..., z_{n-1}) by solving one MIP per
    coordinate; ``bounds`` may contain additional variables (e.g. slack,
    binary-enable or tier variables), which stay free after the first ``n``
    coordinates.  The model's commissioning/tier rows are added on every
    solve.  Returns the full solution vector; callers take ``[: model.n]``
    for the batch-count variables."""
    n = model.n if n_coords is None else n_coords
    bnds = list(model.bounds if bounds is None else bounds)
    n_vars = len(bnds)
    z = np.zeros(n_vars)
    last_x = None
    for k in range(n):
        obj = np.zeros(n_vars)
        obj[k] = 1.0
        r = _lp(obj, model, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bnds)
        if r.status != 0:
            raise RuntimeError(
                f"lexicographic refinement failed at index {k}: {r.message}"
            )
        lo0, hi0 = bnds[k]
        v = min(max(float(r.x[k]), lo0), hi0)
        if model.integrality[k]:
            # Pin integer coordinates at the exact integer returned by the
            # MIP.  A fractional pin (v +/- relative eps) may contain no
            # integer at all once v is large, making the next solve
            # spuriously infeasible.
            iv = int(round(v))
            iv = max(int(math.ceil(lo0 - 1e-9)), min(int(math.floor(hi0 + 1e-9)), iv))
            z[k] = float(iv)
            bnds[k] = (float(iv), float(iv))
        else:
            z[k] = v
            eps = _FIX_RTOL * max(1.0, abs(v))
            bnds[k] = (max(lo0, v - eps), min(hi0, v + eps))
        last_x = r.x
    # Carry the enable/tier/deviation variables of the last solved MIP so the
    # returned vector is a complete feasible, cost-optimal solution.
    if last_x is not None:
        for j in range(n, n_vars):
            z[j] = float(last_x[j])
    return z


# --------------------------------------------------------------------------- #
# supply-reduction contingencies
# --------------------------------------------------------------------------- #

def solve_contingency(
    sources: List[SourceData],
    target_volume: float,
    limits: Mapping[str, Tuple[float, float]],
    original_volumes: Mapping[str, float],
    new_caps: Mapping[str, float],
    minimum_active_sources: Optional[int] = None,
) -> ContingencyOutput:
    """Re-optimize a saved optimal plan under tightened supply caps.

    The lexicographic objective is:

      1. minimize the total absolute volume change
         ``sum_i |x_i - x_i^0|`` versus the stored allocation;
      2. among minimum-change allocations, minimize total cost;
      3. among those, minimize the usage vector (source ids ascending)
         lexicographically.

    Target volume, indicator windows, source concentrations, unit costs and
    tier price blocks, batch sizes, minimum-run thresholds and the
    active-source count are unchanged (the reduced caps clamp every tier
    capacity).  ``new_caps`` only needs to list sources whose cap is
    reduced; every other source keeps its original ``max_volume``.
    """
    sources = sorted(sources, key=lambda s: s.id)
    model = _build_model(
        sources, target_volume, limits, caps=new_caps,
        minimum_active_sources=minimum_active_sources,
    )
    n = model.n
    scales = model.scales
    # The stored plan honors the batch lattice; snap tiny rounding noise so
    # that the L1 reference point sits exactly on it.
    x0 = np.zeros(n)
    for i, s in enumerate(sources):
        v = float(original_volumes[s.id])
        if s.batch_size is not None:
            v = float(round(v / s.batch_size) * s.batch_size)
        x0[i] = v

    # Variables are ordered (z, y, w/q tier vars, d_1..d_n).  Build the
    # zero-padded view over the extended space up front so every phase uses
    # the model's indicator rows, volume equality and commissioning/tier
    # structural rows at the extended width.
    A1, b1, bounds1, d0 = _l1_model(x0, model)
    nv = len(bounds1)
    ext_model = _pad_model(model, nv)

    # Phase 1: minimize sum(d_i) subject to |scale_i z_i - x0_i| <= d_i.
    obj1 = np.zeros(nv)
    obj1[d0:d0 + n] = 1.0
    p1 = _lp(obj1, ext_model, A_ub=A1, b_ub=b1, bounds=bounds1)
    if p1.status == 2:  # provably infeasible under the new caps
        achievable, message = _analyze_infeasible(
            sources, model, target_volume, limits, caps=new_caps
        )
        return ContingencyOutput(
            status="infeasible",
            achievable_ranges=achievable,
            message=message,
        )
    if p1.status != 0:
        raise RuntimeError(f"MIP solver failed (status {p1.status}): {p1.message}")
    adj_star = float(p1.fun)

    c_ext = _cost_row(sources, ext_model, nv)
    z_all = _contingency_lexicographic(
        c_ext, x0, adj_star, A1, b1, bounds1, d0, ext_model,
    )

    z = z_all[:n]
    x = scales * z
    deltas = x - x0
    indicators = {
        name: float(np.array([getattr(s, name) for s in sources]) @ x) / target_volume
        for name in INDICATORS
    }
    total_cost = float(c_ext @ z_all)
    old_cost = float(sum(source_cost(s, x0[i]) for i, s in enumerate(sources)))
    return ContingencyOutput(
        status="optimal",
        volumes=x,
        deltas=deltas,
        total_adjustment=float(np.abs(deltas).sum()),
        total_cost=total_cost,
        cost_delta=total_cost - old_cost,
        indicators=indicators,
    )


def _l1_model(x0, model: _Model):
    """Build the L1 deviation rows and the extended bounds.

    The extended variables are ``(z, y, w/q tier vars, d_1..d_n)``.  The
    returned inequality group contains the indicator-window rows (zero
    padded over the new columns) followed by the deviation rows
    ``|scale_i z_i - x0_i| <= d_i``; the commissioning/tier structural rows
    are appended by ``_lp`` on every solve.
    """
    n = model.n
    nv = model.n_vars + n
    d0 = model.n_vars
    mid = model.n_vars - n  # y + tier columns between z and d
    eye = np.eye(n)
    dev_rows = [
        np.hstack([eye * model.scales.reshape(-1, 1),
                   np.zeros((n, mid)), -eye]),
        np.hstack([-eye * model.scales.reshape(-1, 1),
                   np.zeros((n, mid)), -eye]),
    ]
    dev = np.vstack(dev_rows)
    dev_rhs = np.concatenate([x0, -x0])
    if model.A_ub is not None:
        A = np.vstack([
            np.hstack([model.A_ub, np.zeros((model.A_ub.shape[0], n))]),
            dev,
        ])
        b = np.concatenate([model.b_ub, dev_rhs])
    else:
        A, b = dev, dev_rhs
    bounds_ext = list(model.bounds) + [(0.0, None)] * n
    return A, b, bounds_ext, d0


def _pad_model(model: _Model, n_vars_ext: int) -> _Model:
    """A view of the model whose matrices span the contingency's extended
    variable space (zero-padded over the deviation columns)."""
    def _pad(mat):
        if mat is None:
            return None
        return np.hstack([mat, np.zeros((mat.shape[0], n_vars_ext - model.n_vars))])

    return _Model(
        n=model.n,
        n_vars=model.n_vars,
        scales=model.scales,
        integrality=np.concatenate(
            [model.integrality, np.zeros(n_vars_ext - model.n_vars)]
        ),
        bounds=list(model.bounds) + [(0.0, None)] * (n_vars_ext - model.n_vars),
        A_eq=_pad(model.A_eq),
        b_eq=model.b_eq,
        A_ub=_pad(model.A_ub),
        b_ub=model.b_ub,
        A_struct=_pad(model.A_struct),
        b_struct=model.b_struct,
        A_struct_eq=_pad(model.A_struct_eq),
        b_struct_eq=model.b_struct_eq,
        y_col=model.y_col,
        threshold=model.threshold,
        minimum_active=model.minimum_active,
        minimum_run=model.minimum_run,
        tier_cols=model.tier_cols,
        tier_caps=model.tier_caps,
        tier_prices=model.tier_prices,
    )


def _contingency_lexicographic(
    c, x0, adj_star, A_l1, b_l1, bounds_ext, d0, ext_model: _Model,
) -> np.ndarray:
    """Phase 2+3: within the minimum-change allocations, minimize cost and
    then the id-sorted usage vector lexicographically.  ``ext_model`` is the
    zero-padded view over the extended (z, y, w/q, d) variable space."""
    n = ext_model.n
    nv = len(bounds_ext)
    change_row = np.zeros(nv)
    change_row[d0:d0 + n] = 1.0
    adj_tol = _FIX_RTOL * max(1.0, abs(adj_star))

    # Phase 2: minimum cost with the total change pinned at its minimum.
    A_eq2 = np.vstack([ext_model.A_eq, change_row])
    b_eq2 = np.append(ext_model.b_eq, adj_star)
    p2 = _lp(c, ext_model, A_ub=A_l1, b_ub=b_l1, A_eq=A_eq2, b_eq=b_eq2,
             bounds=bounds_ext)
    if p2.status != 0:  # numerical issue: relax the pin to a tight inequality
        p2 = _lp(
            c, ext_model,
            A_ub=np.vstack([A_l1, change_row]),
            b_ub=np.append(b_l1, adj_star + adj_tol),
            bounds=bounds_ext,
        )
        if p2.status != 0:
            raise RuntimeError(
                f"contingency cost refinement failed (status {p2.status}): {p2.message}"
            )
    cost_star = float(p2.fun)

    # Phase 3: lex-min (z_1, ..., z_n) among min-change, min-cost solutions.
    A_eq3 = np.vstack([A_eq2, c.reshape(1, -1)])
    b_eq3 = np.append(b_eq2, cost_star)
    try:
        return _sequential_min(ext_model, A_l1, b_l1, A_eq3, b_eq3,
                               bounds=bounds_ext)
    except RuntimeError:
        cost_tol = _COST_RTOL * max(1.0, abs(cost_star))
        A_ub3 = np.vstack([A_l1, change_row, c.reshape(1, -1)])
        b_ub3 = np.append(b_l1, [adj_star + adj_tol, cost_star + cost_tol])
        return _sequential_min(ext_model, A_ub3, b_ub3, bounds=bounds_ext)


# --------------------------------------------------------------------------- #
# infeasibility analysis
# --------------------------------------------------------------------------- #

def _infeasible(sources, model, target_volume, limits,
                caps=None) -> SolveOutput:
    """Explain infeasibility with per-indicator achievable ranges."""
    achievable, message = _analyze_infeasible(
        sources, model, target_volume, limits, caps=caps
    )
    return SolveOutput(
        status="infeasible",
        achievable_ranges=achievable,
        message=message,
    )


def _max_attainable_volume(sources, caps, volume=None,
                           minimum_active_sources=None) -> float:
    """Largest volume attainable under the given caps, batch sizes and
    commissioning (minimum-run / active-count) / tier-ordering constraints.

    When ``volume`` is given it acts as an upper bound, so the result is the
    largest attainable volume not exceeding it."""
    m = _build_model(
        sources, 0.0, {}, caps=caps,
        minimum_active_sources=minimum_active_sources,
    )
    A_ub, b_ub = None, None
    if volume is not None:
        row = np.zeros(m.n_vars)
        row[: m.n] = m.scales
        A_ub = row.reshape(1, -1)
        b_ub = np.array([float(volume)])
    c = np.zeros(m.n_vars)
    c[: m.n] = -m.scales
    res = _lp(c, m, A_ub=A_ub, b_ub=b_ub, A_eq=None, b_eq=None)
    if res.status != 0:
        return 0.0
    return float(m.scales @ res.x[: m.n])


def _analyze_infeasible(sources, model, target_volume, limits, caps=None):
    """Per-indicator achievable ranges and a human-readable explanation,
    projected onto the mixed-integer supply/volume feasible set.

    The projection reuses the *same* caps as the failing model (original
    supply caps on the solve path, reduced caps on the contingency path) and
    honors every batch-size, minimum-run, active-source and tier-ordering
    constraint.  When the exact target volume cannot be assembled, the
    ranges are reported at the largest attainable volume."""
    n = model.n
    k_required = model.minimum_active
    z_caps = np.array([hi for _, hi in model.bounds[:n]], dtype=float)
    total_supply = float(sum(hi * scale for hi, scale in zip(z_caps, model.scales)))
    achievable: Dict[str, dict] = {}
    messages: List[str] = []

    # A threshold source can only be enabled when its lattice ceiling of the
    # minimum-run volume still fits under its (possibly reduced) cap.
    cap_volumes = model.scales * z_caps
    eligible = model.threshold & (model.minimum_run <= cap_volumes + 1e-9)
    n_eligible = int(np.count_nonzero(eligible))
    count_impossible = k_required > n_eligible
    if k_required > 0:
        achievable["active_sources"] = {
            "requested": k_required, "available": n_eligible,
        }
        if count_impossible:
            messages.append(
                f"minimum_active_sources={k_required} cannot be met: only "
                f"{n_eligible} source(s) declaring minimum_run_volume can "
                "reach their threshold under the supply caps"
            )

    volume = float(target_volume)
    if total_supply < target_volume * (1.0 - 1e-9):
        messages.append(
            f"total available volume {total_supply:g} is below the target {target_volume:g}"
        )
        achievable["volume"] = {"requested": float(target_volume), "max": float(total_supply)}
        volume = float(total_supply)

    # Aggregate supply may suffice while the batch lattice / minimum-run
    # thresholds cannot produce the exact (or the reduced) volume: test the
    # structural constraints (caps, batches, commissioning, tier linkage; no
    # indicator windows) at the candidate volume, then fall back to the
    # largest attainable volume on the SAME constraint set.  Skip the probe
    # outright when the active-source count is already known unattainable.
    if not count_impossible and volume > 0 and model.integrality.any():
        probe = _build_model(
            sources, volume, {}, caps=caps,
            minimum_active_sources=k_required,
        )
        feas = _lp(np.zeros(probe.n_vars), probe, A_ub=None, b_ub=None)
        if feas.status == 2:
            vmax = _max_attainable_volume(
                sources, caps, volume=volume,
                minimum_active_sources=k_required,
            )
            reasons = []
            if model.integrality[:n].any():
                reasons.append("batch-size")
            if model.threshold.any():
                reasons.append("minimum-run/active-source")
            messages.append(
                f"the target volume {target_volume:g} cannot be assembled under the "
                f"{' and '.join(reasons)} constraints; the largest attainable volume "
                f"not above it is {vmax:g}"
            )
            achievable["volume"] = {
                "requested": float(target_volume), "max": float(vmax),
            }
            volume = float(vmax)
            if volume <= 0.0:
                count_impossible = True
                messages.append(
                    "no positive-volume allocation satisfies the commissioning constraints"
                )

    if volume > 0 and not count_impossible:
        # Projection model: the same caps, batch lattice, commissioning and
        # tier constraints, but indicator windows are deliberately excluded
        # so the min/max MIPs scan the whole structural feasible set.
        proj_model = _build_model(
            sources, volume, {}, caps=caps,
            minimum_active_sources=k_required,
        )
        for name in INDICATORS:
            conc = np.zeros(proj_model.n_vars)
            conc[:n] = (
                np.array([getattr(s, name) for s in sources], dtype=float)
                * proj_model.scales
            )
            lo_res = _lp(conc, proj_model)
            hi_res = _lp(-conc, proj_model)
            entry: Dict[str, float] = {}
            if lo_res.status == 0:
                entry["min"] = float(lo_res.fun) / volume
            if hi_res.status == 0:
                entry["max"] = float(-hi_res.fun) / volume
            req = limits.get(name)
            if req is not None:
                entry["requested_min"], entry["requested_max"] = float(req[0]), float(req[1])
                if "max" in entry and entry["max"] < req[0]:
                    messages.append(
                        f"{name}: requested minimum {req[0]:g} exceeds the achievable "
                        f"maximum {entry['max']:.6g}"
                    )
                if "min" in entry and entry["min"] > req[1]:
                    messages.append(
                        f"{name}: requested maximum {req[1]:g} is below the achievable "
                        f"minimum {entry['min']:.6g}"
                    )
            achievable[name] = entry
    elif not messages:
        messages.append("no feasible allocation exists under the given constraints")

    if not messages:
        messages.append(
            "indicator ranges are individually attainable but jointly incompatible "
            "under the supply, volume, batch-size and commissioning constraints"
        )
    return achievable, "; ".join(messages)
