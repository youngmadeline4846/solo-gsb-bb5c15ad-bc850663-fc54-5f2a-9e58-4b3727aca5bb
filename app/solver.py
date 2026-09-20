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

    minimize    sum(cost_i * x_i)
    subject to  sum(x_i)                       = target_volume
                lo_j * target_volume <= sum(conc_ij * x_i) <= hi_j * target_volume
                x_i                            <= max_volume_i
                z_i in Z>=0 for batched sources, z_i in R>=0 otherwise
                x_i in {0} or [m_i, U_i] for threshold sources (via y_i)
                sum(y_i) >= K when minimum_active_sources is set

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
    cost: float
    calcium: float
    magnesium: float
    alkalinity: float
    batch_size: Optional[float] = None
    minimum_run_volume: Optional[float] = None


@dataclass
class _Model:
    """Constraint matrix description in the (possibly scaled) variable space.

    Volumes are recovered as ``x = scales * z`` for the first ``n`` columns;
    ``integrality`` marks the batch-count variables and the binary enable
    variables.  Columns ``n`` onward hold one binary enable variable per
    threshold source, in id-sorted order (see ``y_col``).

    ``A_struct``/``b_struct`` hold the commissioning (minimum-run and active-
    count) constraints, which are part of *every* projection of the feasible
    set, including the indicator-free infeasibility analysis.  ``A_ub``/
    ``b_ub`` hold the indicator windows, which that analysis deliberately
    drops.
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
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Tuple[float, float]],
           np.ndarray, Dict[int, int], np.ndarray]:
    """Build the minimum-run big-M rows and the active-count row.

    Returns (A_struct, b_struct, extended_bounds, y_col, extended_integrality,
    minimum_run).  Variable layout is (z_0..z_{n-1}, y_0..y_{m-1}) with one
    binary column per threshold source.
    """
    n = len(sources)
    idx = np.flatnonzero(threshold)
    m = idx.size
    n_vars = n + m
    y_col = {int(i): n + k for k, i in enumerate(idx)}
    ext_bounds = list(bounds) + [(0.0, 1.0)] * m
    integ = np.concatenate([_integrality(sources), np.ones(m)])

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
    A_struct = np.vstack(rows) if rows else None
    b_struct = np.asarray(rhs) if rows else None
    return A_struct, b_struct, ext_bounds, y_col, integ, minimum_run


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
    (A_struct, b_struct, ext_bounds, y_col, integ,
     minimum_run) = _commissioning_constraints(
        sources, scales, bounds, threshold, minimum_run, minimum_active
    )
    n_vars = n + int(threshold.sum())
    A_eq = np.concatenate([scales.reshape(1, -1), np.zeros((1, n_vars - n))], axis=1)
    b_eq = np.array([float(target_volume)])
    A_ub, b_ub = _indicator_constraints(
        sources, target_volume, limits, scales, n_vars
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
    costs = np.array([s.cost for s in sources], dtype=float)
    c = np.concatenate([costs * model.scales, np.zeros(model.n_vars - model.n)])

    res = _lp(c, model)
    if res.status == 2:  # provably infeasible
        return _infeasible(sources, model, target_volume, limits)
    if res.status != 0:
        raise RuntimeError(f"MIP solver failed (status {res.status}): {res.message}")

    z = _lexicographic_min(c, res.x, model)
    x = model.scales * z
    indicators = {
        name: float(np.array([getattr(s, name) for s in sources]) @ x) / target_volume
        for name in INDICATORS
    }
    return SolveOutput(
        status="optimal",
        volumes=x,
        total_cost=float(costs @ x),
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

    Target volume, indicator windows, unit costs, batch sizes, minimum-run
    thresholds and the active-source count are unchanged.  ``new_caps`` only
    needs to list sources whose cap is reduced; every other source keeps its
    original ``max_volume``.
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
    costs = np.array([s.cost for s in sources], dtype=float)

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

    c_ext = np.concatenate([costs * scales, np.zeros(nv - n)])
    z = _contingency_lexicographic(
        c_ext, x0, adj_star, A1, b1, A1_eq, b1_eq, bounds1, integ1,
        model,
    )

    x = scales * z
    deltas = x - x0
    indicators = {
        name: float(np.array([getattr(s, name) for s in sources]) @ x) / target_volume
        for name in INDICATORS
    }
    return ContingencyOutput(
        status="optimal",
        volumes=x,
        deltas=deltas,
        total_adjustment=float(np.abs(deltas).sum()),
        total_cost=float(costs @ x),
        cost_delta=float(costs @ (x - x0)),
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
