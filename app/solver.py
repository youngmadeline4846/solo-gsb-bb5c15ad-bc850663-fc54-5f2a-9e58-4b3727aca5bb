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

A source may replace its flat ``cost`` with ``price_tiers``: an ordered list
of (capacity, unit price) segments whose capacities cover ``max_volume``
(the last segment may be open-ended).  Tier-usage variables t_{i,k} sum to
the source volume and are consumed in order — one binary variable per tier
boundary enforces

    t_{i,k} <= c_{i,k} * w_{i,k}          (tier k closed while w = 0)
    t_{i,k-1} >= c_{i,k-1} * w_{i,k}      (tier k opens only once k-1 is full)

so the source cost is the piecewise-linear sum of its tier prices, minimized
in the same MIP together with the batch, commissioning and quality
constraints.

    minimize    sum(cost_i * x_i)  (flat sources) + sum(price_ik * t_ik)  (tiered)
    subject to  sum(x_i)                       = target_volume
                lo_j * target_volume <= sum(conc_ij * x_i) <= hi_j * target_volume
                x_i                            <= max_volume_i
                z_i in Z>=0 for batched sources, z_i in R>=0 otherwise
                x_i in {0} or [m_i, U_i] for threshold sources (via y_i)
                sum(y_i) >= K when minimum_active_sources is set
                sum_k t_{i,k} = x_i and in-order tier usage (via w_ik)

Continuous, batched and commissioned sources are solved together in a single
mixed-integer program; a continuous optimum is never rounded or trimmed
afterwards (small usages are never pruned in a post-processing pass).

When several optima share the same minimal cost, the usage vector ordered by
ascending source id is minimized lexicographically via a sequence of MIPs.

If the problem is infeasible, the achievable (min, max) concentration range of
every indicator is reported, projected onto the *same* mixed-integer feasible
set (supply caps, volume, batch and commissioning constraints).

Supply-reduction contingency plans (``solve_contingency``) use the same
mixed-integer feasible set with tightened supply caps, but optimize a
lexicographic objective: first minimize the total absolute volume change
versus the stored allocation, then minimize cost, and finally the
source-id-ordered usage vector lexicographically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
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
    cost: Optional[float]
    calcium: float
    magnesium: float
    alkalinity: float
    batch_size: Optional[float] = None
    minimum_run_volume: Optional[float] = None
    # Ordered (capacity, unit_price) segments replacing the flat cost; a
    # null capacity on the last segment extends it up to max_volume.
    price_tiers: Optional[List[Tuple[Optional[float], float]]] = None


@dataclass
class _Model:
    """Constraint matrix description in the (possibly scaled) variable space.

    Volumes are recovered as ``x = scales * z`` for the first ``n`` columns;
    ``integrality`` marks the batch-count variables and the binary enable
    variables.  Columns ``n`` onward hold one binary enable variable per
    threshold source, in id-sorted order (see ``y_col``), then the
    tier-usage columns ``t_{i,k}`` (continuous, see ``tier_cols``) and one
    binary tier-boundary variable per tier beyond the first.

    ``A_struct``/``b_struct`` hold the commissioning (minimum-run and active-
    count) constraints and the tier-ordering constraints, which are part of
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
    y_col: Dict[int, int]        # source index -> enable-variable column
    threshold: np.ndarray        # bool per source: declares minimum_run_volume
    minimum_active: int          # required active threshold sources (>= 0)
    minimum_run: np.ndarray      # minimum-run volume per source (0 if unset)
    tier_cols: Dict[int, List[int]]      # source index -> tier-usage columns
    tier_caps: Dict[int, List[float]]    # source index -> effective tier caps
    tier_prices: Dict[int, List[float]]  # source index -> tier unit prices


@dataclass
class SolveOutput:
    status: str  # "optimal" | "infeasible"
    volumes: Optional[np.ndarray] = None       # aligned with the sorted sources
    total_cost: Optional[float] = None
    source_costs: Optional[np.ndarray] = None  # per-source cost, sorted order
    # per source (sorted order): tier usages, or None for flat-cost sources
    tier_volumes: Optional[List[Optional[List[float]]]] = None
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
    source_costs: Optional[np.ndarray] = None  # per-source cost, sorted order
    tier_volumes: Optional[List[Optional[List[float]]]] = None
    indicators: Optional[Dict[str, float]] = None
    achievable_ranges: Optional[Dict[str, dict]] = None
    message: Optional[str] = None


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


def _effective_tiers(s: SourceData) -> List[Tuple[float, float]]:
    """(capacity, unit_price) pairs with the open last tier resolved against
    ``max_volume`` and every tier truncated at the remaining headroom, so the
    effective capacities always sum to ``min(total capacity, max_volume)``."""
    tiers: List[Tuple[float, float]] = []
    acc = 0.0
    for cap, price in s.price_tiers or []:
        headroom = max(0.0, float(s.max_volume) - acc)
        eff = headroom if cap is None else min(float(cap), headroom)
        tiers.append((max(0.0, eff), float(price)))
        acc += tiers[-1][0]
    return tiers


def _tier_usage(eff_tiers: Sequence[Tuple[float, float]], volume: float
                ) -> List[float]:
    """Decompose a source volume into its pricing tiers.

    The in-order constraints make this decomposition unique: tier k is
    filled completely before tier k+1 takes anything.  Any residual is
    solver noise (the MIP itself bounds the volume by the total tier
    capacity) and is folded into the last tier so the usages sum exactly
    to ``volume``.
    """
    vols: List[float] = []
    rem = max(0.0, float(volume))
    for cap, _ in eff_tiers:
        v = min(rem, cap)
        vols.append(v)
        rem -= v
    if rem > 0.0 and vols:
        vols[-1] += rem
    return vols


def _allocation_costs(
    sources: Sequence[SourceData], x: np.ndarray
) -> Tuple[np.ndarray, List[Optional[List[float]]]]:
    """Per-source cost and per-tier usage breakdown of a volume vector."""
    per_source = np.zeros(len(sources))
    tier_volumes: List[Optional[List[float]]] = []
    for i, s in enumerate(sources):
        if s.price_tiers:
            eff = _effective_tiers(s)
            vols = _tier_usage(eff, float(x[i]))
            tier_volumes.append(vols)
            per_source[i] = float(
                sum(v * price for v, (_, price) in zip(vols, eff))
            )
        else:
            tier_volumes.append(None)
            per_source[i] = float(s.cost) * float(x[i])
    return per_source, tier_volumes


def _cost_vector(sources: Sequence[SourceData], model: "_Model") -> np.ndarray:
    """Full-width linear cost vector: flat cost on the z columns of flat
    sources, tier unit prices on the tier-usage columns of tiered sources."""
    c = np.zeros(model.n_vars)
    for i, s in enumerate(sources):
        cols = model.tier_cols.get(i)
        if cols is not None:
            for col, price in zip(cols, model.tier_prices[i]):
                c[col] = price
        else:
            c[i] = (0.0 if s.cost is None else float(s.cost)) * model.scales[i]
    return c


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
    """Indicator-window rows in z-space, zero-padded over the y columns:
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
    n_vars: int,
) -> Tuple[List[np.ndarray], List[float], Dict[int, int], np.ndarray]:
    """Build the minimum-run big-M rows and the active-count row.

    Returns (rows, rhs, y_col, minimum_run).  One binary enable column per
    threshold source sits at ``n + k`` (threshold sources in id-sorted
    order); the rows are built at the full model width so the tier-pricing
    columns can follow the enable columns.
    """
    n = len(sources)
    idx = np.flatnonzero(threshold)
    y_col = {int(i): n + k for k, i in enumerate(idx)}

    rows: List[np.ndarray] = []
    rhs: List[float] = []
    for i in idx:
        k = y_col[int(i)]
        # Largest drawable volume (z-units), floored to the batch lattice.
        u_z = bounds[int(i)][1]
        u_vol = scales[int(i)] * u_z
        # x_i - U_i*y_i <= 0: disabled (y=0) forces x_i = 0.
        row = np.zeros(n_vars)
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
        row = np.zeros(n_vars)
        row[int(i)] = -scales[int(i)]
        row[k] = m_eff
        rows.append(row)
        rhs.append(0.0)
        minimum_run[int(i)] = m_eff
    if minimum_active > 0:
        # -sum(y_i) <= -K: at least K threshold sources must be enabled.
        row = np.zeros(n_vars)
        for i in idx:
            row[y_col[int(i)]] = -1.0
        rows.append(row)
        rhs.append(float(-minimum_active))
    return rows, rhs, y_col, minimum_run


def _tier_constraints(
    scales: np.ndarray,
    tier_caps: Dict[int, List[float]],
    tier_cols: Dict[int, List[int]],
    w_cols: Dict[int, List[int]],
    n_vars: int,
) -> Tuple[List[np.ndarray], List[float], List[np.ndarray], List[float]]:
    """Tier-pricing rows at full model width.

    Equality rows couple the tier usages to the source volume
    (``sum_k t_{i,k} = scale_i z_i``); structural rows enforce in-order
    consumption with one binary w_{i,k} per tier boundary k >= 1:

        t_{i,k}   <= c_{i,k}   * w_{i,k}   (tier k is closed while w = 0)
        t_{i,k-1} >= c_{i,k-1} * w_{i,k}   (tier k opens only once k-1 is full)
    """
    eq_rows: List[np.ndarray] = []
    eq_rhs: List[float] = []
    struct_rows: List[np.ndarray] = []
    struct_rhs: List[float] = []
    for i in sorted(tier_cols):
        cols = tier_cols[i]
        caps = tier_caps[i]
        row = np.zeros(n_vars)
        row[i] = -scales[i]
        for col in cols:
            row[col] = 1.0
        eq_rows.append(row)
        eq_rhs.append(0.0)
        for k, w in enumerate(w_cols.get(i, []), start=1):
            row = np.zeros(n_vars)
            row[cols[k]] = 1.0
            row[w] = -caps[k]
            struct_rows.append(row)
            struct_rhs.append(0.0)
            row = np.zeros(n_vars)
            row[cols[k - 1]] = -1.0
            row[w] = caps[k - 1]
            struct_rows.append(row)
            struct_rhs.append(0.0)
    return eq_rows, eq_rhs, struct_rows, struct_rhs


def _build_model(
    sources: Sequence[SourceData],
    target_volume: float,
    limits: Mapping[str, Tuple[float, float]],
    caps: Optional[Mapping[str, float]] = None,
    minimum_active_sources: Optional[int] = None,
) -> _Model:
    """Assemble the (mixed-integer) feasible-set description in z-space.

    Variable layout: ``(z_1..z_n, y_1..y_m, t..., w...)`` — usage units,
    enable binaries, tier-usage variables and tier-boundary binaries.
    """
    n = len(sources)
    scales = _scales(sources)
    bounds = _source_bounds(sources, scales, caps)
    threshold = _threshold_mask(sources)
    minimum_run = _minimum_run(sources)
    minimum_active = int(minimum_active_sources or 0)
    m = int(threshold.sum())

    # Tiered-pricing layout: effective capacities are resolved against
    # max_volume (contingency caps tighten the z bounds only, never the
    # price segments).
    tier_caps: Dict[int, List[float]] = {}
    tier_prices: Dict[int, List[float]] = {}
    for i, s in enumerate(sources):
        if s.price_tiers:
            eff = _effective_tiers(s)
            tier_caps[i] = [cap for cap, _ in eff]
            tier_prices[i] = [price for _, price in eff]
    n_t = sum(len(v) for v in tier_caps.values())
    n_w = sum(max(0, len(v) - 1) for v in tier_caps.values())
    n_vars = n + m + n_t + n_w

    tier_cols: Dict[int, List[int]] = {}
    w_cols: Dict[int, List[int]] = {}
    col = n + m
    for i in sorted(tier_caps):
        tier_cols[i] = list(range(col, col + len(tier_caps[i])))
        col += len(tier_caps[i])
    for i in sorted(tier_caps):
        k = len(tier_caps[i]) - 1
        if k > 0:
            w_cols[i] = list(range(col, col + k))
            col += k

    struct_rows, struct_rhs, y_col, minimum_run = _commissioning_constraints(
        sources, scales, bounds, threshold, minimum_run, minimum_active, n_vars
    )
    eq_tier, b_eq_tier, ord_rows, ord_rhs = _tier_constraints(
        scales, tier_caps, tier_cols, w_cols, n_vars
    )

    volume_row = np.zeros(n_vars)
    volume_row[:n] = scales
    A_eq = np.vstack([volume_row] + eq_tier)
    b_eq = np.asarray([float(target_volume)] + b_eq_tier, dtype=float)
    A_ub, b_ub = _indicator_constraints(
        sources, target_volume, limits, scales, n_vars
    )
    all_struct = struct_rows + ord_rows
    all_struct_rhs = struct_rhs + ord_rhs
    A_struct = np.vstack(all_struct) if all_struct else None
    b_struct = np.asarray(all_struct_rhs) if all_struct else None
    ext_bounds = (
        list(bounds)
        + [(0.0, 1.0)] * m
        + [(0.0, cap) for i in sorted(tier_caps) for cap in tier_caps[i]]
        + [(0.0, 1.0)] * n_w
    )
    integ = np.concatenate(
        [_integrality(sources), np.ones(m), np.zeros(n_t), np.ones(n_w)]
    )
    return _Model(
        n=n,
        n_vars=n_vars,
        scales=scales,
        integrality=integ,
        bounds=ext_bounds,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        A_struct=A_struct,
        b_struct=b_struct,
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
    """Append the commissioning rows to an inequality group."""
    if model.A_struct is None:
        return A_ub, b_ub
    if A_ub is None:
        return model.A_struct, model.b_struct
    return np.vstack([A_ub, model.A_struct]), np.concatenate([b_ub, model.b_struct])


def _lp(c, model: _Model, bounds=_UNSET, A_ub=_UNSET, b_ub=_UNSET,
        A_eq=_UNSET, b_eq=_UNSET, struct: bool = True):
    """linprog wrapper forwarding the model's variable integrality.

    Constraint arguments default to the model's matrices; pass ``None``
    explicitly to drop that constraint group entirely.  The commissioning
    (minimum-run / active-count) rows are always included unless
    ``struct=False`` (used only by the structural-feasibility probes)."""
    a_ub = model.A_ub if A_ub is _UNSET else A_ub
    b_ubv = model.b_ub if b_ub is _UNSET else b_ub
    if struct:
        a_ub, b_ubv = _stack_struct(a_ub, b_ubv, model)
    return linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ubv,
        A_eq=model.A_eq if A_eq is _UNSET else A_eq,
        b_eq=model.b_eq if b_eq is _UNSET else b_eq,
        bounds=model.bounds if bounds is _UNSET else bounds,
        integrality=model.integrality,
        method="highs",
    )


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
    c = _cost_vector(sources, model)

    res = _lp(c, model)
    if res.status == 2:  # provably infeasible
        return _infeasible(sources, model, target_volume, limits)
    if res.status != 0:
        raise RuntimeError(f"MIP solver failed (status {res.status}): {res.message}")

    z = _lexicographic_min(c, res.x, model)
    x = model.scales * z
    source_costs, tier_volumes = _allocation_costs(sources, x)
    indicators = {
        name: float(np.array([getattr(s, name) for s in sources]) @ x) / target_volume
        for name in INDICATORS
    }
    return SolveOutput(
        status="optimal",
        volumes=x,
        total_cost=float(source_costs.sum()),
        source_costs=source_costs,
        tier_volumes=tier_volumes,
        indicators=indicators,
    )


def _lexicographic_min(c, z_star, model: _Model) -> np.ndarray:
    """Among all cost-optimal solutions, minimize (z_1, ..., z_n)
    lexicographically (indices follow the id-sorted source order; enable
    variables follow after).  Scaling is non-negative so this also
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
    coordinate; ``bounds`` may contain additional variables (e.g. slack or
    binary-enable variables), which stay free after the first ``n``
    coordinates.  The model's commissioning rows are added on every solve."""
    n = model.n if n_coords is None else n_coords
    bnds = list(model.bounds if bounds is None else bounds)
    n_vars = len(bnds)
    z = np.zeros(n_vars)
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
    return z[: model.n]


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

    Target volume, indicator windows, unit costs / price tiers, batch
    sizes, minimum-run thresholds and the active-source count are
    unchanged.  ``new_caps`` only needs to list sources whose cap is
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
    cost_vec = _cost_vector(sources, model)

    # Phase 1: minimize sum(d_i) subject to |scale_i z_i - x0_i| <= d_i.
    # Variables are ordered (z_1..z_n, y_1..y_m, d_1..d_n).
    A1, b1, A1_eq, b1_eq, bounds1, integ1 = _l1_model(x0, model)
    nv = len(bounds1)
    p1 = linprog(
        np.concatenate([np.zeros(nv - n), np.ones(n)]),
        A_ub=A1, b_ub=b1, A_eq=A1_eq, b_eq=b1_eq,
        bounds=bounds1, integrality=integ1, method="highs",
    )
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

    c_ext = np.concatenate([cost_vec, np.zeros(nv - model.n_vars)])
    z = _contingency_lexicographic(
        c_ext, x0, adj_star, A1, b1, A1_eq, b1_eq, bounds1, integ1,
        model,
    )

    x = scales * z
    deltas = x - x0
    source_costs, tier_volumes = _allocation_costs(sources, x)
    original_costs, _ = _allocation_costs(sources, x0)
    indicators = {
        name: float(np.array([getattr(s, name) for s in sources]) @ x) / target_volume
        for name in INDICATORS
    }
    return ContingencyOutput(
        status="optimal",
        volumes=x,
        deltas=deltas,
        total_adjustment=float(np.abs(deltas).sum()),
        total_cost=float(source_costs.sum()),
        cost_delta=float((source_costs - original_costs).sum()),
        source_costs=source_costs,
        tier_volumes=tier_volumes,
        indicators=indicators,
    )


def _l1_model(x0, model: _Model):
    """Build the (z, y, d) model whose feasible set is exactly the allocations
    z (with enable variables y) satisfying the original constraints with
    |scale_i z_i - x0_i| <= d_i."""
    n = model.n
    m = model.n_vars - n
    nv = model.n_vars + n
    eye = np.eye(n)
    rows = [
        np.hstack([eye * model.scales.reshape(-1, 1),
                   np.zeros((n, m)), -eye]),  # scale z - d <= x0
        np.hstack([-eye * model.scales.reshape(-1, 1),
                   np.zeros((n, m)), -eye]),  # -scale z - d <= -x0
    ]
    rhs = [x0, -x0]
    base_rows = []
    base_rhs = []
    if model.A_ub is not None:
        base_rows.append(np.hstack(
            [model.A_ub, np.zeros((model.A_ub.shape[0], n))]))
        base_rhs.append(model.b_ub)
    if model.A_struct is not None:
        base_rows.append(np.hstack(
            [model.A_struct, np.zeros((model.A_struct.shape[0], n))]))
        base_rhs.append(model.b_struct)
    if base_rows:
        rows = base_rows + rows
        rhs = base_rhs + rhs
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    A_eq_ext = np.hstack([model.A_eq, np.zeros((model.A_eq.shape[0], n))])
    bounds_ext = list(model.bounds) + [(0.0, None)] * n
    integ_ext = np.concatenate([model.integrality, np.zeros(n)])
    return A, b, A_eq_ext, model.b_eq, bounds_ext, integ_ext


def _contingency_lexicographic(
    c, x0, adj_star, A_l1, b_l1, A_eq_ext, b_eq_ext, bounds_ext, integ_ext,
    model: _Model,
) -> np.ndarray:
    """Phase 2+3: within the minimum-change allocations, minimize cost and
    then the id-sorted usage vector lexicographically."""
    n = model.n
    nv = len(bounds_ext)
    change_row = np.concatenate([np.zeros(nv - n), np.ones(n)])
    cost_row = c
    adj_tol = _FIX_RTOL * max(1.0, abs(adj_star))

    # Phase 2: minimum cost with the total change pinned at its minimum.
    A_eq2 = np.vstack([A_eq_ext, change_row])
    b_eq2 = np.append(b_eq_ext, adj_star)
    p2 = linprog(
        cost_row, A_ub=A_l1, b_ub=b_l1, A_eq=A_eq2, b_eq=b_eq2,
        bounds=bounds_ext, integrality=integ_ext, method="highs",
    )
    if p2.status != 0:  # numerical issue: relax the pin to a tight inequality
        p2 = linprog(
            cost_row,
            A_ub=np.vstack([A_l1, change_row]),
            b_ub=np.append(b_l1, adj_star + adj_tol),
            A_eq=A_eq_ext, b_eq=b_eq_ext, bounds=bounds_ext,
            integrality=integ_ext, method="highs",
        )
        if p2.status != 0:
            raise RuntimeError(
                f"contingency cost refinement failed (status {p2.status}): {p2.message}"
            )
    cost_star = float(p2.fun)

    # Phase 3: lex-min (z_1, ..., z_n) among min-change, min-cost solutions.
    A_eq3 = np.vstack([A_eq2, cost_row])
    b_eq3 = np.append(b_eq2, cost_star)

    def _pad(mat):
        if mat is None:
            return None
        return np.hstack([mat, np.zeros((mat.shape[0], nv - model.n_vars))])

    ext_model = _Model(
        n=n,
        n_vars=model.n_vars,
        scales=model.scales,
        integrality=integ_ext,
        bounds=bounds_ext,
        A_eq=_pad(model.A_eq),
        b_eq=model.b_eq,
        A_ub=_pad(model.A_ub),
        b_ub=model.b_ub,
        A_struct=_pad(model.A_struct),
        b_struct=model.b_struct,
        y_col=model.y_col,
        threshold=model.threshold,
        minimum_active=model.minimum_active,
        minimum_run=model.minimum_run,
        tier_cols=model.tier_cols,
        tier_caps=model.tier_caps,
        tier_prices=model.tier_prices,
    )
    try:
        return _sequential_min(ext_model, A_l1, b_l1, A_eq3, b_eq3)
    except RuntimeError:
        cost_tol = _COST_RTOL * max(1.0, abs(cost_star))
        A_ub3 = np.vstack([A_l1, change_row, cost_row])
        b_ub3 = np.append(b_l1, [adj_star + adj_tol, cost_star + cost_tol])
        return _sequential_min(ext_model, A_ub3, b_ub3, A_eq_ext, b_eq_ext)


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
    commissioning (minimum-run / active-count) constraints.

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
    honors every batch-size, minimum-run and active-source constraint.  When
    the exact target volume cannot be assembled, the ranges are reported at
    the largest attainable volume."""
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
    # structural constraints (caps, batches, commissioning; no indicator
    # windows) at the candidate volume, then fall back to the largest
    # attainable volume on the SAME constraint set.  Skip the probe outright
    # when the active-source count is already known to be unattainable.
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
        # Projection model: the same caps, batch lattice and commissioning
        # constraints, but indicator windows are deliberately excluded so the
        # min/max MIPs scan the whole structural feasible set.
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
