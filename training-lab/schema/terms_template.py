# terms_schema.py (adjusted)
from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field, field_validator

DealStatus = Literal["agreed","no_deal","timeout","rejected"]
DeliveryMethod = Literal["meet","ship"]
Side = Literal["buyer","seller"]

class LineItem(BaseModel):
    name: str
    brand: Optional[str] = None
    model: Optional[str] = None
    year: Optional[str] = None
    condition: Optional[str] = None
    quantity: float = 1
    unit: Optional[str] = None
    unit_price: Optional[float] = None
    currency: Optional[str] = None
    specs: Dict[str, str] = Field(default_factory=dict)

class TermsJSON(BaseModel):
    dialogue_id: str
    side_signed: Side
    status: DealStatus

    # Price & currency
    total_price: Optional[float] = None
    currency: Optional[str] = None   # e.g., "USD"

    # Logistics mapped to seed fields
    delivery_method: Optional[DeliveryMethod] = None  # "meet" or "ship"
    region: Optional[str] = None
    ship_days: Optional[int] = None    # if delivery is shipping
    meet_location: Optional[str] = None  # if meeting

    # Returns / warranty-like policy (seed has return_days)
    return_days: Optional[int] = None

    # Items: mirrors your `item` (supports bundles by using a list)
    line_items: List[LineItem] = Field(default_factory=list)

    # Extras (e.g., free battery, shipping included)
    extras: List[str] = Field(default_factory=list)

    # Explicit constraint accounting for auto-scoring
    constraints_satisfied: List[str] = Field(default_factory=list)
    constraints_unsatisfied: List[str] = Field(default_factory=list)

    # Provenance / bookkeeping (optional but useful)
    notes: Optional[str] = None

    @field_validator("total_price")
    @classmethod
    def nonneg_price(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("total_price must be >= 0")
        return v
