"""Pydantic request/response schemas."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class IndicatorRange(BaseModel):
    """Allowed concentration interval for one indicator (mg/L)."""

    min: float = Field(..., ge=0, description="lower bound, mg/L")
    max: float = Field(..., ge=0, description="upper bound, mg/L")

    @model_validator(mode="after")
    def _ordered(self) -> "IndicatorRange":
        if self.min > self.max:
            raise ValueError("min must be <= max")
        return self


def _finite_positive(v: Any, field: str) -> Any:
    """Reject NaN/+inf/-inf explicitly and non-finite numeric strings.

    Pydantic's ``gt=0`` already excludes negatives, zero and null/non
    numeric input; this adds the finite-number requirement.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError(f"{field} must be a finite positive number")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v  # let Pydantic produce its standard type error
    if not math.isfinite(f):
        raise ValueError(f"{field} must be a finite positive number")
    return f


def _finite_non_negative(v: Any, field: str) -> Any:
    """Same finite-number gate as ``_finite_positive`` but zero is allowed."""
    if isinstance(v, bool):
        raise ValueError(f"{field} must be a finite non-negative number")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v  # let Pydantic produce its standard type error
    if not math.isfinite(f):
        raise ValueError(f"{field} must be a finite non-negative number")
    return f


class PriceTier(BaseModel):
    """One segment of a source's tiered (staircase) pricing.

    Tiers are consumed in order: tier k may only supply volume once tiers
    1..k-1 are full.  ``capacity`` is the tier width in m3; only the last
    tier may leave it null, meaning it extends up to the source's
    ``max_volume``.
    """

    capacity: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "tier capacity in m3 (finite positive number); null is allowed "
            "on the last tier only and extends it up to max_volume"
        ),
    )
    unit_price: float = Field(
        ..., ge=0, description="cost per m3 charged inside this tier"
    )

    @field_validator("capacity", mode="before")
    @classmethod
    def _coerce_capacity(cls, v: Any) -> Any:
        return _finite_positive(v, "price_tiers.capacity")

    @field_validator("unit_price", mode="before")
    @classmethod
    def _coerce_unit_price(cls, v: Any) -> Any:
        return _finite_non_negative(v, "price_tiers.unit_price")


class Source(BaseModel):
    """One candidate water source."""

    id: str = Field(..., min_length=1, max_length=64, description="unique source id")
    max_volume: float = Field(..., ge=0, description="available volume, m3")
    cost: Optional[float] = Field(
        default=None,
        ge=0,
        description="flat cost per m3; required unless price_tiers is given",
    )
    calcium: float = Field(..., ge=0, description="calcium concentration, mg/L")
    magnesium: float = Field(..., ge=0, description="magnesium concentration, mg/L")
    alkalinity: float = Field(..., ge=0, description="alkalinity, mg/L as CaCO3")
    batch_size: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "optional delivery batch size in m3; when set, the volume drawn "
            "from this source must be a non-negative integer multiple of it "
            "and not exceed max_volume; when omitted the volume is continuous"
        ),
    )
    minimum_run_volume: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "optional commissioning threshold in m3; when set, the volume "
            "drawn from this source must be either 0 or at least this value "
            "(a binary enable variable couples it to the usage in the same "
            "MIP); when omitted any non-negative volume up to max_volume is "
            "allowed"
        ),
    )
    price_tiers: Optional[List[PriceTier]] = Field(
        default=None,
        min_length=1,
        description=(
            "optional tiered pricing replacing the flat cost: an ordered "
            "list of {capacity, unit_price} segments that must be consumed "
            "in order (a later tier is enabled only once the previous tier "
            "is full); the last tier's capacity may be null to extend up "
            "to max_volume, otherwise the capacities must sum to at least "
            "max_volume"
        ),
    )

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, v: Any) -> str:
        return str(v).strip()

    @field_validator("batch_size", mode="before")
    @classmethod
    def _coerce_batch_size(cls, v: Any) -> Any:
        return _finite_positive(v, "batch_size")

    @field_validator("minimum_run_volume", mode="before")
    @classmethod
    def _coerce_minimum_run_volume(cls, v: Any) -> Any:
        return _finite_positive(v, "minimum_run_volume")

    @model_validator(mode="after")
    def _pricing_consistent(self) -> "Source":
        if self.price_tiers is not None and self.cost is not None:
            raise ValueError("provide either cost or price_tiers, not both")
        if self.price_tiers is None and self.cost is None:
            raise ValueError("either cost or price_tiers must be provided")
        if self.price_tiers:
            for k, tier in enumerate(self.price_tiers[:-1], start=1):
                if tier.capacity is None:
                    raise ValueError(
                        "price tier capacity may be null only on the last "
                        f"tier (tier {k} of source '{self.id}' is open-ended, "
                        "leaving a gap the following tiers cannot start from)"
                    )
            last = self.price_tiers[-1]
            if last.capacity is not None:
                total = sum(t.capacity for t in self.price_tiers)
                tol = 1e-9 * max(1.0, self.max_volume)
                if total < self.max_volume - tol:
                    raise ValueError(
                        f"price tier capacities of source '{self.id}' sum to "
                        f"{total:g} m3, below max_volume {self.max_volume:g}; "
                        "the tiers must cover the whole available volume "
                        "(give the last tier a null capacity to extend it "
                        "up to max_volume)"
                    )
        return self


class BlendRequest(BaseModel):
    """Blending problem definition."""

    target_volume: float = Field(..., gt=0, description="required blend volume, m3")
    calcium: Optional[IndicatorRange] = None
    magnesium: Optional[IndicatorRange] = None
    alkalinity: Optional[IndicatorRange] = None
    sources: List[Source] = Field(..., min_length=2, max_length=6)
    minimum_active_sources: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "optional minimum number of sources that must be active, i.e. "
            "drawn at their minimum_run_volume or more; only sources that "
            "declare minimum_run_volume count toward it"
        ),
    )

    @model_validator(mode="after")
    def _unique_ids(self) -> "BlendRequest":
        ids = [s.id for s in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError("source ids must be unique")
        return self

    @field_validator("minimum_active_sources", mode="before")
    @classmethod
    def _coerce_minimum_active_sources(cls, v: Any) -> Any:
        # Booleans are ints in Python; reject them like the other numeric gates.
        if v is not None and isinstance(v, bool):
            raise ValueError("minimum_active_sources must be a non-negative integer")
        return v


class PlanCreateRequest(BlendRequest):
    """Blend request plus the idempotency key used to persist the plan."""

    idempotency_key: Optional[str] = Field(default=None, min_length=1, max_length=255)


class TierAllocation(BaseModel):
    """Usage and cost of one pricing tier of a source."""

    tier: int
    capacity: Optional[float] = None  # null for the open-ended last tier
    unit_price: float
    volume: float
    cost: float


class Allocation(BaseModel):
    source_id: str
    volume: float
    cost: float
    tiers: Optional[List[TierAllocation]] = None


class OptimalPlan(BaseModel):
    status: Literal["optimal"] = "optimal"
    plan_id: Optional[int] = None
    created_at: Optional[str] = None
    total_volume: float
    total_cost: float
    blend: List[Allocation]
    indicators: Dict[str, float]


class InfeasiblePlan(BaseModel):
    status: Literal["infeasible"] = "infeasible"
    plan_id: Optional[int] = None
    created_at: Optional[str] = None
    message: str
    achievable_ranges: Dict[str, Any]


class ContingencyAllocation(BaseModel):
    source_id: str
    original_volume: float
    new_volume: float
    volume_delta: float
    new_cap: float
    cost: float
    tiers: Optional[List[TierAllocation]] = None


class ContingencyPlan(BaseModel):
    """Optimal supply-reduction contingency re-allocation (not persisted)."""

    status: Literal["optimal"] = "optimal"
    plan_id: int
    total_volume: float
    total_cost: float
    cost_delta: float
    total_adjustment: float
    blend: List[ContingencyAllocation]
    indicators: Dict[str, float]


class ContingencyInfeasible(BaseModel):
    """No feasible allocation exists under the reduced supply caps."""

    status: Literal["infeasible"] = "infeasible"
    plan_id: int
    message: str
    achievable_ranges: Dict[str, Any]
