#!/usr/bin/env python3
"""
Quick test to verify the prompt format fix.
This tests that the model responds correctly to User/Assistant format.
"""

from custom_eval.model_wrapper import SimpleModelWrapper
from custom_eval.prompt_builder import PromptBuilder
import logging

logging.basicConfig(level=logging.INFO)

def test_prompt_format():
    """Test that the model responds correctly with User/Assistant format."""

    print("\n" + "="*80)
    print("TESTING PROMPT FORMAT FIX")
    print("="*80)

    # Load model with your checkpoint
    model = SimpleModelWrapper(
        model_name="mistralai/Mistral-7B-v0.1",
        adapter_path="models/diagnostic_500_1/checkpoint-500"
    )

    prompt_builder = PromptBuilder()

    # Test 1: Initial buyer prompt (buyer starts as Assistant)
    print("\n--- Test 1: Buyer starts negotiation ---")
    buyer_prompt = prompt_builder.build_initial_prompt(
        role='buyer',
        item={'name': 'Honda Accord', 'list_price': 25000},
        role_params={'max_price': 23000, 'target_price': 22000}
    )

    print("PROMPT:")
    print(buyer_prompt[-200:])  # Show last 200 chars

    response = model.generate(
        buyer_prompt,
        temperature=0.7,
        max_tokens=100,
        stop_sequences=["\n\n", "User:", "Assistant:", "user:", "assistant:"]
    )

    print("\nGENERATED RESPONSE:")
    print(response[:200])  # Show first 200 chars

    # Check for problematic patterns
    bad_patterns = ["Buyer:", "Seller:", "###", "Summary:", "Lessons Learned:"]
    problems_found = []
    for pattern in bad_patterns:
        if pattern in response:
            problems_found.append(pattern)

    if problems_found:
        print(f"\n❌ PROBLEM: Found unwanted patterns: {problems_found}")
    else:
        print("\n✓ No problematic patterns found!")

    # Test 2: Conversation with history
    print("\n--- Test 2: Ongoing conversation ---")
    conversation_history = [
        ('buyer', "I'm interested in your Honda Accord. Would you take $22,000?"),
        ('seller', "Thanks for your offer, but I'm asking $25,000. Can you go higher?")
    ]

    buyer_prompt2 = prompt_builder.build_initial_prompt(
        role='buyer',
        item={'name': 'Honda Accord', 'list_price': 25000},
        role_params={'max_price': 23000, 'target_price': 22000},
        conversation_history=conversation_history
    )

    print("PROMPT WITH HISTORY:")
    print(buyer_prompt2[-300:])  # Show last 300 chars

    response2 = model.generate(
        buyer_prompt2,
        temperature=0.7,
        max_tokens=100,
        stop_sequences=["\n\n", "User:", "Assistant:", "user:", "assistant:"]
    )

    print("\nGENERATED RESPONSE:")
    print(response2[:200])

    # Check response quality
    problems_found2 = []
    for pattern in bad_patterns:
        if pattern in response2:
            problems_found2.append(pattern)

    if problems_found2:
        print(f"\n❌ PROBLEM: Found unwanted patterns: {problems_found2}")
    else:
        print("\n✓ No problematic patterns found!")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

    return len(problems_found) == 0 and len(problems_found2) == 0

if __name__ == "__main__":
    success = test_prompt_format()
    if success:
        print("\n✓ Format fix appears to be working!")
        print("You can now run the full evaluation:")
        print("  python custom_eval/run_selfchat.py --adapter-path models/diagnostic_500_1/checkpoint-500 --num-seeds 5")
    else:
        print("\n❌ Issues detected. The model may still be generating synthetic dialogs.")
        print("Consider additional fixes or retraining with aligned formats.")