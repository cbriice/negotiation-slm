#!/usr/bin/env python3
"""
Test script for Claude key strategy labeler.
Tests basic functionality with a small sample.
"""

import json
import os
from pathlib import Path
from scripts.claude_key_strategy_labeler import ClaudeStrategyLabeler


def create_test_data():
    """Create a small test dataset."""
    test_packet = {
        "messages": [
            {"role": "system", "content": "You are an expert negotiator."},
            {"role": "assistant", "content": "I need this item and my budget is strictly $100. That's the absolute maximum I can pay."},
            {"role": "user", "content": "I understand your position, but the item is worth $150. Can we meet somewhere in the middle?"},
            {"role": "assistant", "content": "I appreciate your flexibility, but $100 is truly all I have. I cannot go any higher."},
            {"role": "user", "content": "I see. Well, the lowest I can go is $120."},
            {"role": "assistant", "content": "$100 is my final offer. Take it or leave it."}
        ],
        "system": {
            "assistant_style": "non-cooperation",
            "difficulty": 0.4,
            "key_strategy": ""
        },
        "meta": {
            "opponent_style": "compromise",
            "favorable_outcome": "neutral",
            "deal_closed": "false",
            "rounds": 5
        }
    }

    return test_packet


def test_labeler():
    """Test the labeler with a sample packet."""
    print("=== Testing Claude Key Strategy Labeler ===\n")

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found in environment.")
        print("Please set your API key in .env file or environment variables.")
        return

    # Show model being used
    model = os.getenv("CLAUDE_MODEL") or "claude-3-haiku-20240307"
    print(f"Testing with model: {model}\n")

    try:
        # Initialize labeler
        labeler = ClaudeStrategyLabeler()

        # Create test packet
        test_packet = create_test_data()

        print("Test negotiation:")
        print("-" * 50)
        for msg in test_packet["messages"]:
            if msg["role"] != "system":
                print(f"{msg['role'].upper()}: {msg['content']}")
        print("-" * 50)

        print("\nProcessing packet...")

        # Process the packet
        result = labeler.process_packet(test_packet)

        # Check result
        strategy = result.get("system", {}).get("key_strategy", "")

        if strategy:
            print(f"\nSuccess! Detected strategy: '{strategy}'")
            print("\nAnalysis: This negotiation shows the 'anchor' strategy - ")
            print("the assistant stands firm at $100 and refuses to budge.")
        else:
            print("\nFailed to get strategy label")

        # Show stats
        print("\n" + "=" * 50)
        labeler.print_stats()

    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    test_labeler()