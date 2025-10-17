"""
Price Dynamics Module for Negotiation Simulations.

This module implements the Faratin exponential concession model for price adjustments
during negotiations. It handles price changes based on personality parameters and
negotiation progression.
"""

import math
import random
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PriceState:
    """Tracks current price state in negotiation."""
    buyer_price: float  # Current buyer offer
    seller_price: float  # Current seller ask
    initial_buyer_price: float  # Initial buyer offer
    initial_seller_price: float  # Initial seller ask
    current_price: float  # The current negotiated price (latest offer)
    turn_number: int  # Current turn number


class PriceDynamics:
    """Manages price concession dynamics during negotiation."""

    def __init__(self):
        """Initialize price dynamics manager."""
        self.price_history = []

    def calculate_initial_buyer_offer(self, seller_min: float, alpha: float) -> float:
        """
        Calculate the initial price offer from buyer.

        Args:
            seller_min: Seller's minimum acceptable price
            alpha: Buyer personality parameter (typically 0.5-1.0)

        Returns:
            Initial buyer offer price
        """
        return seller_min * alpha

    def calculate_faratin_concession(self,
                                    current_own: float,
                                    current_other: float,
                                    previous_own: float,
                                    previous_other: float,
                                    c: float,
                                    k: int,
                                    is_buyer: bool) -> float:
        """
        Calculate price concession using Faratin exponential model.

        From documentation:
        P_s.k = P_b.k-1 + (P_s.k-1 - P_b.k-1) * exp(-c*k)
        P_b.k = P_s.k-1 + (P_b.k-1 - P_s.k-1) * exp(-c*k)

        Args:
            current_own: Current price for this party
            current_other: Current price for other party
            previous_own: Previous price for this party
            previous_other: Previous price for other party
            c: Convergence speed parameter from personality
            k: Turn number
            is_buyer: Whether calculating for buyer

        Returns:
            New price after concession
        """
        if is_buyer:
            # Buyer increases price towards seller
            # P_b.k = P_s.k-1 + (P_b.k-1 - P_s.k-1) * exp(-c*k)
            new_price = previous_other + (previous_own - previous_other) * math.exp(-c * k)
        else:
            # Seller decreases price towards buyer
            # P_s.k = P_b.k-1 + (P_s.k-1 - P_b.k-1) * exp(-c*k)
            new_price = previous_other + (previous_own - previous_other) * math.exp(-c * k)

        return new_price

    def should_concede_price(self, p_param: float) -> bool:
        """
        Determine if price should be conceded based on personality parameter.

        Args:
            p_param: Probability of price concession from personality

        Returns:
            True if price should be conceded
        """
        return random.random() < p_param

    def calculate_price_adjustment(self,
                                  state: PriceState,
                                  personality,
                                  is_buyer: bool) -> Tuple[float, bool]:
        """
        Calculate the price adjustment for current turn.

        Args:
            state: Current price state
            personality: Personality object with traits including p and c parameters
            is_buyer: Whether calculating for buyer

        Returns:
            Tuple of (new_price, was_conceded)
        """
        # Check if price should be conceded
        if not self.should_concede_price(personality.traits.p):
            # No concession - keep current price
            if is_buyer:
                return state.buyer_price, False
            else:
                return state.seller_price, False

        # Calculate concession using Faratin model
        if is_buyer:
            new_price = self.calculate_faratin_concession(
                current_own=state.buyer_price,
                current_other=state.seller_price,
                previous_own=state.buyer_price,
                previous_other=state.seller_price,
                c=personality.traits.c,
                k=state.turn_number,
                is_buyer=True
            )

            # Ensure buyer doesn't exceed their maximum
            # (personality.traits.beta adjusts the threshold)
            max_price = state.initial_seller_price  # Could be adjusted by buyer_max from seed
            new_price = min(new_price, max_price)

        else:
            new_price = self.calculate_faratin_concession(
                current_own=state.seller_price,
                current_other=state.buyer_price,
                previous_own=state.seller_price,
                previous_other=state.buyer_price,
                c=personality.traits.c,
                k=state.turn_number,
                is_buyer=False
            )

            # Ensure seller doesn't go below their minimum
            # (personality.traits.beta adjusts the threshold)
            min_price = state.initial_buyer_price  # Could be adjusted by seller_min from seed
            new_price = max(new_price, min_price)

        # Round to reasonable decimal places
        new_price = round(new_price, 2)

        # Store in history
        self.price_history.append({
            'turn': state.turn_number,
            'is_buyer': is_buyer,
            'old_price': state.buyer_price if is_buyer else state.seller_price,
            'new_price': new_price,
            'conceded': True
        })

        return new_price, True

    def update_state_with_new_price(self, state: PriceState, new_price: float,
                                   is_buyer: bool) -> PriceState:
        """
        Update price state with new offer.

        Args:
            state: Current price state
            new_price: New price being offered
            is_buyer: Whether this is a buyer offer

        Returns:
            Updated price state
        """
        if is_buyer:
            state.buyer_price = new_price
        else:
            state.seller_price = new_price

        state.current_price = new_price
        return state

    def calculate_price_gap(self, state: PriceState) -> float:
        """
        Calculate the gap between buyer and seller prices.

        Args:
            state: Current price state

        Returns:
            Absolute price gap
        """
        return abs(state.seller_price - state.buyer_price)

    def is_deal_possible(self, state: PriceState,
                        buyer_max: float,
                        seller_min: float) -> bool:
        """
        Check if a deal is possible given current prices and limits.

        Args:
            state: Current price state
            buyer_max: Maximum price buyer will pay
            seller_min: Minimum price seller will accept

        Returns:
            True if deal is possible
        """
        # Deal is possible if buyer's current offer >= seller's min
        # and seller's current ask <= buyer's max
        return (state.buyer_price >= seller_min and
                state.seller_price <= buyer_max)

    def get_middle_ground_price(self, state: PriceState) -> float:
        """
        Calculate a middle ground price between current offers.

        Args:
            state: Current price state

        Returns:
            Middle ground price
        """
        return (state.buyer_price + state.seller_price) / 2

    def format_price_for_prompt(self, price: float) -> str:
        """
        Format price for insertion into prompts.

        Args:
            price: Price value

        Returns:
            Formatted price string
        """
        # Format as currency with 2 decimal places
        return f"${price:.2f}"

    def reset(self):
        """Reset price dynamics for new negotiation."""
        self.price_history = []