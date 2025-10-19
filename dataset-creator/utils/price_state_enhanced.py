"""
Enhanced Price State for separate buyer/seller price tracking.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EnhancedPriceState:
    """Track buyer and seller prices separately throughout negotiation."""

    # Current offers from each party
    current_buyer_offer: float      # Most recent price buyer proposed
    current_seller_offer: float     # Most recent price seller proposed

    # Initial offers (for calculating concession)
    initial_buyer_offer: float      # First buyer price offer
    initial_seller_offer: float     # First seller price offer (list price)

    # Best offers made (closest to agreement)
    best_buyer_offer: float         # Highest price buyer has offered
    best_seller_offer: float        # Lowest price seller has offered

    # Metadata
    turn_number: int
    last_mover: Optional[str] = None  # "BUYER" or "SELLER" - who made last price move

    @property
    def current_gap(self) -> float:
        """Current gap between buyer and seller offers."""
        return abs(self.current_seller_offer - self.current_buyer_offer)

    @property
    def gap_percentage(self) -> float:
        """Gap as percentage of seller's current offer."""
        if self.current_seller_offer > 0:
            return (self.current_gap / self.current_seller_offer) * 100
        return 0

    @property
    def convergence_ratio(self) -> float:
        """How much the gap has closed (0 = no progress, 1 = converged)."""
        initial_gap = abs(self.initial_seller_offer - self.initial_buyer_offer)
        if initial_gap > 0:
            return 1 - (self.current_gap / initial_gap)
        return 1.0

    def get_midpoint(self) -> float:
        """Get midpoint between current offers."""
        return (self.current_buyer_offer + self.current_seller_offer) / 2

    def is_overlapping(self) -> bool:
        """Check if buyer offer exceeds seller offer (deal zone)."""
        return self.current_buyer_offer >= self.current_seller_offer

    def get_active_price_for(self, is_buyer: bool) -> float:
        """Get the current active price for buyer or seller."""
        return self.current_buyer_offer if is_buyer else self.current_seller_offer

    def get_other_party_price(self, is_buyer: bool) -> float:
        """Get the other party's current price."""
        return self.current_seller_offer if is_buyer else self.current_buyer_offer

    def update_price(self, new_price: float, is_buyer: bool) -> Tuple[bool, float]:
        """
        Update price for one party.

        Returns:
            Tuple of (was_concession, concession_amount)
        """
        if is_buyer:
            old_price = self.current_buyer_offer
            self.current_buyer_offer = new_price
            self.best_buyer_offer = max(self.best_buyer_offer, new_price)
            self.last_mover = "BUYER"
            # For buyer, concession means increasing price
            was_concession = new_price > old_price
            concession_amount = new_price - old_price if was_concession else 0
        else:
            old_price = self.current_seller_offer
            self.current_seller_offer = new_price
            self.best_seller_offer = min(self.best_seller_offer, new_price)
            self.last_mover = "SELLER"
            # For seller, concession means decreasing price
            was_concession = new_price < old_price
            concession_amount = old_price - new_price if was_concession else 0

        return was_concession, concession_amount

    def get_favorable_assessment(self, final_price: float,
                                 buyer_max: float, seller_min: float) -> Tuple[bool, bool]:
        """
        Assess if deal is favorable for each party.

        Returns:
            (buyer_favorable, seller_favorable)
        """
        # Calculate zones
        midpoint = (buyer_max + seller_min) / 2
        buyer_good_zone = (seller_min + midpoint) / 2  # Lower 25% of ZOPA
        seller_good_zone = (midpoint + buyer_max) / 2  # Upper 25% of ZOPA

        buyer_favorable = final_price <= buyer_good_zone
        seller_favorable = final_price >= seller_good_zone

        return buyer_favorable, seller_favorable