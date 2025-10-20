"""
Prompt Manager for Negotiation Simulations.

This module handles prompt selection using cosine similarity between personality
trait vectors and prompt embeddings. It manages prompt pools, deduplication,
and forced responses according to negotiation flow.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from prompts.prompts_price_adjust import BuyerPriceAdjustmentPrompts, SellerPriceAdjustmentPrompts
from prompts.prompts_extras import BuyerExtrasPrompts, SellerExtrasPrompts
from prompts.prompts_clarify import BuyerClarifyPrompts, SellerClarifyPrompts
from prompts.prompts_close import BuyerClosePrompts, SellerClosePrompts
from prompts.prompts_accept import BuyerAcceptPrompts, SellerAcceptPrompts
from prompts.prompts_reject import BuyerRejectPrompts, SellerRejectPrompts
from prompts.prompts_ultimatum import BuyerUltimatumPrompts, SellerUltimatumPrompts
from prompts.prompts_unified import GeneralPrompts


@dataclass
class PromptPool:
    """Container for prompts of a specific action type."""
    prompts: List[Dict]
    used_prompts: Set[str]

    def get_available(self) -> List[Dict]:
        """Get prompts that haven't been used yet."""
        return [p for p in self.prompts if p["prompt"] not in self.used_prompts]

    def mark_used(self, prompt: str):
        """Mark a prompt as used."""
        self.used_prompts.add(prompt)


class PromptManager:
    """Manages prompt selection for negotiation simulations."""

    def __init__(self):
        """Initialize prompt manager with all prompt categories."""
        # Initialize buyer prompts
        self.buyer_prompts = {
            "price_adjust": self._load_price_prompts(is_buyer=True),
            "extras": self._load_extras_prompts(is_buyer=True),
            "clarify": self._load_clarify_prompts(is_buyer=True),
            "close": self._load_close_prompts(is_buyer=True),
            "accept": self._load_accept_prompts(is_buyer=True),
            "reject": self._load_reject_prompts(is_buyer=True),
            "ultimatum": self._load_ultimatum_prompts(is_buyer=True)
        }

        # Initialize seller prompts
        self.seller_prompts = {
            "price_adjust": self._load_price_prompts(is_buyer=False),
            "extras": self._load_extras_prompts(is_buyer=False),
            "clarify": self._load_clarify_prompts(is_buyer=False),
            "close": self._load_close_prompts(is_buyer=False),
            "accept": self._load_accept_prompts(is_buyer=False),
            "reject": self._load_reject_prompts(is_buyer=False),
            "ultimatum": self._load_ultimatum_prompts(is_buyer=False)
        }

        # General prompts (not personality-specific)
        self.general_prompts = GeneralPrompts()

        # Track used prompts per negotiation
        self.used_prompts_buyer: Set[str] = set()
        self.used_prompts_seller: Set[str] = set()

    def _load_price_prompts(self, is_buyer: bool) -> Dict[str, PromptPool]:
        """Load price adjustment prompts categorized by personality traits."""
        if is_buyer:
            prompts_obj = BuyerPriceAdjustmentPrompts()
        else:
            prompts_obj = SellerPriceAdjustmentPrompts()

        return {
            "dapc": PromptPool(prompts_obj.dapc_prompts, set()),
            "pc": PromptPool(prompts_obj.pc_prompts, set()),
            "wpc": PromptPool(prompts_obj.wpc_prompts, set()),
            "pa": PromptPool(prompts_obj.pa_prompts, set())
        }

    def _load_extras_prompts(self, is_buyer: bool) -> PromptPool:
        """Load extras negotiation prompts."""
        if is_buyer:
            prompts_obj = BuyerExtrasPrompts()
        else:
            prompts_obj = SellerExtrasPrompts()
        return PromptPool(prompts_obj.prompts, set())

    def _load_clarify_prompts(self, is_buyer: bool) -> PromptPool:
        """Load clarification prompts."""
        if is_buyer:
            prompts_obj = BuyerClarifyPrompts()
        else:
            prompts_obj = SellerClarifyPrompts()
        return PromptPool(prompts_obj.prompts, set())

    def _load_close_prompts(self, is_buyer: bool) -> PromptPool:
        """Load deal closing prompts."""
        if is_buyer:
            prompts_obj = BuyerClosePrompts()
        else:
            prompts_obj = SellerClosePrompts()
        return PromptPool(prompts_obj.prompts, set())

    def _load_accept_prompts(self, is_buyer: bool) -> Dict[str, PromptPool]:
        """Load acceptance prompts (favorable and unfavorable)."""
        if is_buyer:
            prompts_obj = BuyerAcceptPrompts()
        else:
            prompts_obj = SellerAcceptPrompts()

        return {
            "favorable": PromptPool(prompts_obj.prompts, set()),
            "unfavorable": PromptPool(prompts_obj.ufa_prompts, set())
        }

    def _load_reject_prompts(self, is_buyer: bool) -> PromptPool:
        """Load rejection prompts."""
        if is_buyer:
            prompts_obj = BuyerRejectPrompts()
        else:
            prompts_obj = SellerRejectPrompts()
        return PromptPool(prompts_obj.prompts, set())

    def _load_ultimatum_prompts(self, is_buyer: bool) -> PromptPool:
        """Load ultimatum prompts."""
        if is_buyer:
            prompts_obj = BuyerUltimatumPrompts()
        else:
            prompts_obj = SellerUltimatumPrompts()
        return PromptPool(prompts_obj.prompts, set())

    def reset_negotiation(self):
        """Reset used prompts for a new negotiation."""
        self.used_prompts_buyer.clear()
        self.used_prompts_seller.clear()

        # Reset all prompt pools
        for action_pools in self.buyer_prompts.values():
            if isinstance(action_pools, dict):
                for pool in action_pools.values():
                    pool.used_prompts.clear()
            else:
                action_pools.used_prompts.clear()

        for action_pools in self.seller_prompts.values():
            if isinstance(action_pools, dict):
                for pool in action_pools.values():
                    pool.used_prompts.clear()
            else:
                action_pools.used_prompts.clear()

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def select_prompt_by_similarity(self, prompts: List[Dict], personality_vector: np.ndarray,
                                   used_prompts: Set[str]) -> Optional[Tuple[str, float]]:
        """
        Select the best prompt based on cosine similarity with personality vector.

        Returns:
            Tuple of (prompt_text, similarity_score) or None if no prompts available
        """
        available_prompts = [p for p in prompts if p["prompt"] not in used_prompts]

        if not available_prompts:
            return None

        best_prompt = None
        best_score = -1

        for prompt_dict in available_prompts:
            similarity = self.cosine_similarity(personality_vector, prompt_dict["embedding"])
            if similarity > best_score:
                best_score = similarity
                best_prompt = prompt_dict["prompt"]

        return (best_prompt, best_score) if best_prompt else None

    def get_price_adjust_prompt(self, personality, eta: float, cg: float, cz: float,
                               is_buyer: bool, current_price: float) -> str:
        """
        Get appropriate price adjustment prompt based on control features and personality.

        Args:
            personality: Personality object with trait_vector
            eta: Ultimatum parameter from personality
            cg: Control feature for price gap
            cz: Control feature for price range
            is_buyer: Whether this is for buyer or seller
            current_price: Current negotiation price

        Returns:
            Selected prompt with price formatted
        """
        prompts_dict = self.buyer_prompts["price_adjust"] if is_buyer else self.seller_prompts["price_adjust"]
        used_prompts = self.used_prompts_buyer if is_buyer else self.used_prompts_seller

        # Determine prompt category based on control features
        if eta > 0.7 and (cg > 0 or cz > 0):
            pool = prompts_dict["dapc"]
        elif 0.25 <= eta <= 0.7 and (cg > 0 or cz > 0):
            pool = prompts_dict["pc"]
        elif eta < 0.25:
            pool = prompts_dict["wpc"]
        else:
            pool = prompts_dict["pa"]

        # Select best prompt by similarity
        result = self.select_prompt_by_similarity(
            pool.prompts,
            personality.trait_vector,
            used_prompts
        )

        if result:
            prompt, _ = result
            used_prompts.add(prompt)
            return prompt.format(price=current_price)

        # Fallback to any available price prompt
        for _, pool in prompts_dict.items():
            result = self.select_prompt_by_similarity(
                pool.prompts,
                personality.trait_vector,
                used_prompts
            )
            if result:
                prompt, _ = result
                used_prompts.add(prompt)
                return prompt.format(price=current_price)

        # Last resort: generic prompt
        return f"Adjust the price to {current_price}"

    def get_action_prompt(self, action: str, personality, is_buyer: bool,
                         **kwargs) -> str:
        """
        Get prompt for a specific action based on personality.

        Args:
            action: Action type (extras, clarify, close, accept, reject)
            personality: Personality object with trait_vector
            is_buyer: Whether this is for buyer or seller
            **kwargs: Additional parameters for specific actions
                - price: Current price for the action
                - accept_price: Price being accepted (for accept action)
                - buyer_max: Buyer's maximum price (for accept favorability)
                - seller_min: Seller's minimum price (for accept favorability)

        Returns:
            Selected prompt string
        """
        prompts_dict = self.buyer_prompts if is_buyer else self.seller_prompts
        used_prompts = self.used_prompts_buyer if is_buyer else self.used_prompts_seller

        # Special handling for price_adjust
        if action == "price_adjust":
            return self.get_price_adjust_prompt(
                personality,
                kwargs.get("eta", 0.5),
                kwargs.get("cg", 0),
                kwargs.get("cz", 0),
                is_buyer,
                kwargs.get("price", 0)
            )

        # Special handling for accept action - determine favorable vs unfavorable
        if action == "accept":
            prompt_pool = prompts_dict.get(action)
            if isinstance(prompt_pool, dict):
                # Determine if the accept is favorable or unfavorable
                accept_price = kwargs.get("accept_price", kwargs.get("price", 0))

                if is_buyer:
                    # For buyer: favorable if price < buyer_max, unfavorable if price >= buyer_max
                    buyer_max = kwargs.get("buyer_max", float('inf'))
                    pool_key = "favorable" if accept_price < buyer_max else "unfavorable"
                else:
                    # For seller: favorable if price > seller_min, unfavorable if price <= seller_min
                    seller_min = kwargs.get("seller_min", 0)
                    pool_key = "favorable" if accept_price > seller_min else "unfavorable"

                # Get the appropriate pool
                selected_pool = prompt_pool[pool_key]
                all_prompts = selected_pool.prompts
            else:
                # Fallback if not properly structured - should not happen given our initialization
                all_prompts = []
                print(f'[ERROR] problem initializing prompt pool for accept action, prompt_pool is type {type(prompt_pool)}. utils/prompt_manager.py:get_action_prompt()')
        else:
            # Get prompt pool for action
            prompt_pool = prompts_dict.get(action)
            if not prompt_pool:
                return f"Perform {action} action"

            # Handle dictionary of pools (like price_adjust subcategories)
            if isinstance(prompt_pool, dict):
                # For other actions that might have subcategories
                all_prompts = []
                for pool in prompt_pool.values():
                    all_prompts.extend(pool.prompts)
            else:
                all_prompts = prompt_pool.prompts

        # Select best prompt by similarity
        result = self.select_prompt_by_similarity(
            all_prompts,
            personality.trait_vector,
            used_prompts
        )

        if result:
            prompt, _ = result
            used_prompts.add(prompt)

            # Format prompt if it has placeholders
            if "price" in kwargs and "{price}" in prompt:
                return prompt.format(price=kwargs["price"])
            return prompt

        # Fallback
        return f"Perform {action} action"

    def get_ultimatum_prompt(self, personality, is_buyer: bool, current_price: float) -> str:
        """
        Get ultimatum prompt when conditions are met (A_k >= 0.7, appropriate eta).

        Args:
            personality: Personality object
            is_buyer: Whether this is for buyer or seller
            current_price: Current price for ultimatum

        Returns:
            Ultimatum prompt string
        """
        prompts_dict = self.buyer_prompts["ultimatum"] if is_buyer else self.seller_prompts["ultimatum"]
        used_prompts = self.used_prompts_buyer if is_buyer else self.used_prompts_seller

        result = self.select_prompt_by_similarity(
            prompts_dict.prompts,
            personality.trait_vector,
            used_prompts
        )

        if result:
            prompt, _ = result
            used_prompts.add(prompt)
            return prompt.format(price=current_price)

        # Fallback ultimatum
        if is_buyer:
            return f"This is my final offer: {current_price}. Take it or leave it."
        else:
            return f"My final price is {current_price}. I won't go any lower."

    def get_forced_response_prompt(self, response_type: str, is_buyer: bool) -> str:
        """
        Get forced response prompts for structured conversation flow.

        Args:
            response_type: Type of forced response (opener, clarify_respond)
            is_buyer: Whether this is for buyer or seller

        Returns:
            Appropriate general prompt
        """
        if response_type == "opener":
            return self.general_prompts.buyer_opener if is_buyer else self.general_prompts.seller_opener
        elif response_type == "clarify_respond":
            return self.general_prompts.buyer_clarify_respond if is_buyer else self.general_prompts.seller_clarify_respond

        return "Respond appropriately to the partner's message."

    def should_add_ultimatum_to_pool(self, eta: float, A_k: float) -> bool:
        """
        Determine if ultimatum prompts should be added to the pool.

        Args:
            eta: Ultimatum parameter from personality
            A_k: Round pressure from control function

        Returns:
            True if ultimatum should be available
        """
        return eta > 0.5 and A_k >= 0.7