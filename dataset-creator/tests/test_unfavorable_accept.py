#!/usr/bin/env python3
"""
Test script to verify unfavorable accept prompts are working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.prompt_manager import PromptManager
from prompts.personality_deck import PersonalityDeck
import numpy as np

def test_unfavorable_accept_prompts():
    """Test that unfavorable accept prompts are selected correctly."""

    # Initialize prompt manager and personality deck
    prompt_manager = PromptManager()
    personality_deck = PersonalityDeck()

    # Get a random personality for testing
    buyer_personality, seller_personality = personality_deck.select_two_personalities()

    print("Testing Unfavorable Accept Prompts")
    print("=" * 50)

    # Test 1: Buyer accepting at buyer_max (unfavorable for buyer)
    print("\nTest 1: Buyer accepting at buyer_max (should be UNFAVORABLE)")
    buyer_max = 1000
    accept_price = 1000  # Equal to buyer_max, so unfavorable

    prompt = prompt_manager.get_action_prompt(
        "accept",
        buyer_personality,
        is_buyer=True,
        accept_price=accept_price,
        buyer_max=buyer_max,
        seller_min=700
    )
    print(f"  Accept price: ${accept_price}, Buyer max: ${buyer_max}")
    print(f"  Selected prompt: {prompt}")

    # Verify it's an unfavorable prompt by checking key words
    unfavorable_keywords = ["begrudgingly", "stretch", "not ideal", "isn't ideal", "waste more time", "need the item"]
    is_unfavorable = any(keyword in prompt.lower() for keyword in unfavorable_keywords)
    print(f"  Contains unfavorable language: {is_unfavorable}")

    # Test 2: Buyer accepting below buyer_max (favorable for buyer)
    print("\nTest 2: Buyer accepting below buyer_max (should be FAVORABLE)")
    accept_price = 800  # Well below buyer_max, so favorable

    prompt = prompt_manager.get_action_prompt(
        "accept",
        buyer_personality,
        is_buyer=True,
        accept_price=accept_price,
        buyer_max=buyer_max,
        seller_min=700
    )
    print(f"  Accept price: ${accept_price}, Buyer max: ${buyer_max}")
    print(f"  Selected prompt: {prompt}")

    favorable_keywords = ["pleased", "great", "grateful", "happily", "smooth", "amicably"]
    is_favorable = any(keyword in prompt.lower() for keyword in favorable_keywords)
    print(f"  Contains favorable language: {is_favorable}")

    # Test 3: Seller accepting at seller_min (unfavorable for seller)
    print("\nTest 3: Seller accepting at seller_min (should be UNFAVORABLE)")
    seller_min = 700
    accept_price = 700  # Equal to seller_min, so unfavorable

    prompt = prompt_manager.get_action_prompt(
        "accept",
        seller_personality,
        is_buyer=False,
        accept_price=accept_price,
        buyer_max=1000,
        seller_min=seller_min
    )
    print(f"  Accept price: ${accept_price}, Seller min: ${seller_min}")
    print(f"  Selected prompt: {prompt}")

    is_unfavorable = any(keyword in prompt.lower() for keyword in unfavorable_keywords)
    print(f"  Contains unfavorable language: {is_unfavorable}")

    # Test 4: Seller accepting above seller_min (favorable for seller)
    print("\nTest 4: Seller accepting above seller_min (should be FAVORABLE)")
    accept_price = 900  # Well above seller_min, so favorable

    prompt = prompt_manager.get_action_prompt(
        "accept",
        seller_personality,
        is_buyer=False,
        accept_price=accept_price,
        buyer_max=1000,
        seller_min=seller_min
    )
    print(f"  Accept price: ${accept_price}, Seller min: ${seller_min}")
    print(f"  Selected prompt: {prompt}")

    is_favorable = any(keyword in prompt.lower() for keyword in favorable_keywords)
    print(f"  Contains favorable language: {is_favorable}")

    print("\n" + "=" * 50)
    print("Test complete!")

    # Verify the accept prompts are loaded as dictionaries
    print("\nVerifying accept prompt structure:")
    print(f"  Buyer accept prompts type: {type(prompt_manager.buyer_prompts['accept'])}")
    print(f"  Seller accept prompts type: {type(prompt_manager.seller_prompts['accept'])}")

    if isinstance(prompt_manager.buyer_prompts['accept'], dict):
        print(f"  Buyer accept keys: {list(prompt_manager.buyer_prompts['accept'].keys())}")
        print(f"  Favorable prompts: {len(prompt_manager.buyer_prompts['accept']['favorable'].prompts)}")
        print(f"  Unfavorable prompts: {len(prompt_manager.buyer_prompts['accept']['unfavorable'].prompts)}")

    if isinstance(prompt_manager.seller_prompts['accept'], dict):
        print(f"  Seller accept keys: {list(prompt_manager.seller_prompts['accept'].keys())}")
        print(f"  Favorable prompts: {len(prompt_manager.seller_prompts['accept']['favorable'].prompts)}")
        print(f"  Unfavorable prompts: {len(prompt_manager.seller_prompts['accept']['unfavorable'].prompts)}")

if __name__ == "__main__":
    test_unfavorable_accept_prompts()