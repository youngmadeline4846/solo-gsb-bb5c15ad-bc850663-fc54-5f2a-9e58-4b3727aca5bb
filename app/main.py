"""HTTP API for the drinking-water blending service."""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import db
from .config import get_host, get_port
from .models import (
    Allocation,
    BlendRequest,
    ContingencyInfeasible,
    ContingencyPlan,
    InfeasiblePlan,
    OptimalPlan,
    PlanCreateRequest,
)
from .solver import INDICATORS, SourceData, solve_blend, solve_contingency

SolveResponse = Union[OptimalPlan, InfeasiblePlan]
ContingencyResponse = Union[ContingencyPlan, ContingencyInfeasible]


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()  # create the SQLite database/schema on startup
    yield


app = FastAPI(
    title="Drinking Water Blending API",
    version="1.0.0",
    description=(
        "最低成本饮用水混配优化：在满足钙 / 镁 / 碱度区间与水源供给上限的"
        "前提下最小化总成本；水源可设置 batch_size（批次体积），设置后用量"
        "必须为其非负整数倍；水源还可设置 minimum_run_volume（投运门槛），"
        "设置后其用量只能为 0 或不低于该门槛（由二元启用变量在同一 MIP 中"
        "联立）；请求可设置 minimum_active_sources，要求至少该数量的门槛"
        "水源投运。连续、批次与投运约束在同一个混合整数模型中求解；成本"
        "并列时取按水源 ID 排序后用量向量字典序最小者；不可行时返回各指标"
        "在同一混合整数可行集上的可达范围。"
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _round6(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _round6(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round6(v) for v in obj]
    return obj


def _json_safe(obj: Any) -> Any:
    """Make validation-error payloads JSON-serializable.

    Pydantic echoes the offending input (e.g. a NaN/inf ``batch_size``) and
    embeds the original exception object in ``ctx``; neither can be encoded
    by the strict JSON encoder.  Non-finite numbers are replaced by null and
    any other non-primitive value by its string form.
    """
    if isinstance(obj, bool) or isinstance(obj, int) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": _json_safe(exc.errors())},
    )


def _source_data(req: BlendRequest) -> List[SourceData]:
    return sorted(
        (
            SourceData(
                id=s.id,
                max_volume=s.max_volume,
                cost=s.cost,
                calcium=s.calcium,
                magnesium=s.magnesium,
                alkalinity=s.alkalinity,
                batch_size=s.batch_size,
                minimum_run_volume=s.minimum_run_volume,
            )
            for s in req.sources
        ),
        key=lambda s: s.id,
    )


def _limits_from_request(req: BlendRequest) -> Dict[str, Tuple[float, float]]:
    limits = {}
    for name in INDICATORS:
        rng = getattr(req, name)
        if rng is not None:
            limits[name] = (rng.min, rng.max)
    return limits


def _solve(req: BlendRequest) -> Tuple[List[SourceData], Any]:
    sources = _source_data(req)
    limits = _limits_from_request(req)
    try:
        result = solve_blend(
            sources, req.target_volume, limits,
            minimum_active_sources=req.minimum_active_sources,
        )
    except RuntimeError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))
    return sources, result


def _build_body(sources, req: BlendRequest, result, plan_id, created_at) -> Dict[str, Any]:
    if result.status == "optimal":
        body = OptimalPlan(
            plan_id=plan_id,
            created_at=created_at,
            total_volume=req.target_volume,
            total_cost=result.total_cost,
            blend=[
                Allocation(source_id=s.id, volume=v, cost=v * s.cost)
                for s, v in zip(sources, result.volumes)
            ],
            indicators=result.indicators,
        )
    else:
        body = InfeasiblePlan(
            plan_id=plan_id,
            created_at=created_at,
            message=result.message,
            achievable_ranges=result.achievable_ranges,
        )
    return _round6(body.model_dump())


def _contingency_body(
    sources: List[SourceData],
    plan_id: int,
    original: Dict[str, float],
    new_caps: Dict[str, float],
    result,
    target_volume: float,
) -> Dict[str, Any]:
    if result.status == "optimal":
        body = ContingencyPlan(
            plan_id=plan_id,
            total_volume=target_volume,
            total_cost=result.total_cost,
            cost_delta=result.cost_delta,
            total_adjustment=result.total_adjustment,
            blend=[
                {
                    "source_id": s.id,
                    "original_volume": original[s.id],
                    "new_volume": float(v),
                    "volume_delta": float(v - original[s.id]),
                    "new_cap": float(new_caps.get(s.id, s.max_volume)),
                    "cost": float(v * s.cost),
                }
                for s, v in zip(sources, result.volumes)
            ],
            indicators=result.indicators,
        )
    else:
        body = ContingencyInfeasible(
            plan_id=plan_id,
            message=result.message,
            achievable_ranges=result.achievable_ranges,
        )
    return _round6(body.model_dump())


def _canonical_request(req: PlanCreateRequest) -> Tuple[str, str]:
    """Canonical JSON of the blending parameters (key excluded) + its hash."""
    payload = req.model_dump(mode="json")
    payload.pop("idempotency_key", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest(), canonical


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #

@app.get("/")
def root() -> Dict[str, str]:
    return {
        "service": "drinking-water-blending-api",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/blends/solve", response_model=SolveResponse)
def solve_only(req: BlendRequest) -> SolveResponse:
    """Solve a blending problem without persisting anything."""
    sources, result = _solve(req)
    return _build_body(sources, req, result, plan_id=None, created_at=None)


@app.post("/api/v1/blends", response_model=SolveResponse)
def create_plan(
    req: PlanCreateRequest,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    """Solve and persist a plan, idempotently.

    The idempotency key may be supplied either in the request body
    (``idempotency_key``) or via the ``Idempotency-Key`` header.  Replaying a
    key with the identical request returns the stored plan; replaying it with
    a different request fails with 409.
    """
    key = req.idempotency_key or idempotency_key
    if not key:
        raise HTTPException(
            status_code=422,
            detail=(
                "an idempotency key is required: set 'idempotency_key' in the "
                "request body or send the 'Idempotency-Key' header"
            ),
        )
    req_hash, canonical = _canonical_request(req)

    existing = db.find_by_key(key)
    if existing is not None:
        if existing["request_hash"] != req_hash:
            raise HTTPException(
                status_code=409,
                detail="idempotency key was already used with a different request",
            )
        return JSONResponse(
            json.loads(existing["response_json"]),
            headers={"X-Idempotent-Replay": "true"},
        )

    sources, result = _solve(req)
    body = _build_body(sources, req, result, plan_id=None, created_at=_utcnow())
    created, _, stored, existing_hash = db.save_plan(
        key, req_hash, canonical, body["status"], body
    )
    if not created:  # lost a concurrent-insert race on the same key
        if existing_hash != req_hash:
            raise HTTPException(
                status_code=409,
                detail="idempotency key was already used with a different request",
            )
        return JSONResponse(stored, headers={"X-Idempotent-Replay": "true"})
    return JSONResponse(stored, status_code=201)


@app.get("/api/v1/blends/{plan_id}", response_model=SolveResponse)
def get_plan(plan_id: int) -> JSONResponse:
    """Fetch a previously saved plan by its id."""
    row = db.find_by_id(plan_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"plan {plan_id} not found")
    return JSONResponse(json.loads(row["response_json"]))


@app.post(
    "/api/v1/blends/{plan_id}/contingency",
    response_model=ContingencyResponse,
)
def plan_contingency(plan_id: int, reductions: Dict[str, float]) -> JSONResponse:
    """Re-optimize a saved optimal plan under reduced supply caps.

    The request body maps existing source ids to their new supply caps.
    At least one source must be listed, each new cap must satisfy
    ``0 <= new_cap <= original max_volume``, and sources not listed keep
    their original cap.  The target volume, indicator windows, water
    quality (source concentrations) and costs are taken from the stored
    request.  Batch sizes, minimum-run thresholds and minimum_active_sources
    are likewise preserved.  Among feasible allocations the service first
    minimizes the total absolute volume change versus the saved blend, then
    the total cost, and finally takes the id-sorted usage-vector
    lexicographic minimum.

    The contingency result is neither saved nor does it modify the
    original plan.  Infeasibility under the new caps is reported with
    ``200`` and ``"status": "infeasible"``.
    """
    row = db.find_by_id(plan_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"plan {plan_id} not found")
    if row["status"] != "optimal":
        raise HTTPException(
            status_code=422,
            detail=(
                f"plan {plan_id} has status '{row['status']}'; contingencies can "
                "only be derived from plans with status 'optimal'"
            ),
        )
    if not reductions:
        raise HTTPException(
            status_code=422,
            detail="at least one source id with a new supply cap is required",
        )

    req = BlendRequest.model_validate(json.loads(row["request_json"]))
    sources = _source_data(req)
    by_id = {s.id: s for s in sources}

    unknown = sorted(set(reductions) - set(by_id))
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"unknown source id(s): {', '.join(unknown)}; only sources of "
            "the saved plan may be listed",
        )
    bad = [
        sid for sid, cap in reductions.items()
        if not math.isfinite(cap) or cap < 0
    ]
    if bad:
        raise HTTPException(
            status_code=422,
            detail=f"new caps must be finite and non-negative; bad values for: "
            f"{', '.join(sorted(bad))}",
        )
    exceeded = [
        sid for sid, cap in reductions.items()
        if cap > float(by_id[sid].max_volume)
    ]
    if exceeded:
        raise HTTPException(
            status_code=422,
            detail="new cap must not exceed the original max_volume: "
            + ", ".join(
                f"{sid} (original max {by_id[sid].max_volume:g})"
                for sid in sorted(exceeded)
            ),
        )

    stored = json.loads(row["response_json"])
    original = {a["source_id"]: float(a["volume"]) for a in stored["blend"]}

    try:
        result = solve_contingency(
            sources,
            req.target_volume,
            _limits_from_request(req),
            original,
            {sid: float(cap) for sid, cap in reductions.items()},
            minimum_active_sources=req.minimum_active_sources,
        )
    except RuntimeError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(
        _contingency_body(
            sources,
            plan_id,
            original,
            {sid: float(cap) for sid, cap in reductions.items()},
            result,
            req.target_volume,
        )
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=get_host(), port=get_port())
