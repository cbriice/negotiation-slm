#!/usr/bin/env python3
"""
Test script to verify curriculum masking implementation.
Tests masking_regions-based weighting.
"""

import torch
from transformers import AutoTokenizer
from training.curriculum.masking_collator import CurriculumMaskingCollator


def test_next_utterance_masking():
    """Test masking_regions-based weighting for next-utterance format."""
    print("\n" + "="*80)
    print("TEST 1: Next-Utterance Format (masking_regions)")
    print("="*80)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    # Create masking collator with curriculum config
    collator = CurriculumMaskingCollator(tokenizer, "phase1")

    # Simulate curriculum phase config with weighted masking
    phase_config = {
        "masking": {
            "negotiation": {
                "user_weight": 0.05,      # Context weight
                "assistant_weight": 1.0,  # Target weight
                "system_weight": 0.0
            }
        }
    }
    collator.set_phase("phase1", phase_config)

    # Sample text: conversation with 2 assistant turns (early + final)
    text = """---
system: You are a negotiation assistant.
---
User: I want to buy this for $100.
Assistant: I can offer $150.
User: How about $120?
Assistant: Deal! $120 works for me."""

    # Create masking_regions (as created by reformat_to_next_utterance.py)
    masking_regions = [
        {"start": 0, "end": 44, "mask": True, "role": "system"},
        {"start": 44, "end": 80, "mask": True, "role": "user"},
        {"start": 80, "end": 106, "mask": True, "role": "assistant"},  # Non-final (context)
        {"start": 106, "end": 125, "mask": True, "role": "user"},
        {"start": 125, "end": 156, "mask": False, "role": "assistant"}  # Final (target)
    ]

    # Tokenize
    tokenized = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = tokenized["input_ids"][0]

    # Create example
    example = {
        "input_ids": input_ids,
        "text": text,
        "source": "negotiation",
        "masking_regions": masking_regions
    }

    # Apply masking
    result = collator.mask_example(example)
    labels = result["labels"]
    loss_weights = result["loss_weights"]

    # Verify results
    print(f"Text length: {len(text)} chars")
    print(f"Token count: {len(input_ids)} tokens")
    print(f"\nExpected weighting:")
    print(f"  System section (chars 0-44): weight = 0.0")
    print(f"  User turn 1 (chars 44-80): weight = 0.05")
    print(f"  Assistant turn 1 (chars 80-106) [CONTEXT]: weight = 0.05")
    print(f"  User turn 2 (chars 106-125): weight = 0.05")
    print(f"  Assistant turn 2 (chars 125-156) [TARGET]: weight = 1.0")

    # Check unique weights
    unique_weights = torch.unique(loss_weights)
    print(f"\nActual unique weights in tensor: {unique_weights.tolist()}")

    # Count tokens by weight
    weight_counts = {}
    for w in unique_weights:
        count = (loss_weights == w).sum().item()
        weight_counts[w.item()] = count

    print(f"\nWeight distribution:")
    for weight, count in sorted(weight_counts.items()):
        print(f"  Weight {weight:.2f}: {count} tokens")

    # Verify that we have both context weight (0.05) and target weight (1.0)
    assert 0.05 in unique_weights, "Missing context weight (0.05)"
    assert 1.0 in unique_weights, "Missing target weight (1.0)"

    # Verify final assistant turn has weight 1.0
    final_assistant_start = text.find("Assistant: Deal!")
    final_token_pos = len(tokenizer.encode(text[:final_assistant_start], add_special_tokens=False))
    assert loss_weights[final_token_pos] == 1.0, f"Final assistant turn should have weight 1.0, got {loss_weights[final_token_pos]}"

    print("\n✓ Next-utterance masking PASSED")
    return True



def test_dynamic_curriculum_weights():
    """Test that curriculum weights are applied dynamically."""
    print("\n" + "="*80)
    print("TEST 3: Dynamic Curriculum Weights")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    collator = CurriculumMaskingCollator(tokenizer, "phase1")

    text = """---
system: Test
---
User: Hi
Assistant: Hello"""

    masking_regions = [
        {"start": 0, "end": 19, "mask": True, "role": "system"},
        {"start": 19, "end": 29, "mask": True, "role": "user"},
        {"start": 29, "end": 46, "mask": False, "role": "assistant"}
    ]

    tokenized = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = tokenized["input_ids"][0]

    example = {
        "input_ids": input_ids,
        "text": text,
        "source": "negotiation",
        "masking_regions": masking_regions
    }

    # Test Phase 1 weights (high context exposure)
    print("\nPhase 1: user_weight=0.05")
    phase1_config = {
        "masking": {
            "negotiation": {
                "user_weight": 0.05,
                "assistant_weight": 1.0,
                "system_weight": 0.0
            }
        }
    }
    collator.set_phase("phase1", phase1_config)
    result1 = collator.mask_example(example.copy())
    weights1 = torch.unique(result1["loss_weights"])
    print(f"Weights: {weights1.tolist()}")
    assert 0.05 in weights1, "Phase 1 should have user_weight=0.05"

    # Test Phase 2 weights (reduced context exposure)
    print("\nPhase 2: user_weight=0.02")
    phase2_config = {
        "masking": {
            "negotiation": {
                "user_weight": 0.02,
                "assistant_weight": 1.0,
                "system_weight": 0.0
            }
        }
    }
    collator.set_phase("phase2", phase2_config)
    result2 = collator.mask_example(example.copy())
    weights2 = torch.unique(result2["loss_weights"])
    print(f"Weights: {weights2.tolist()}")
    assert 0.02 in weights2, "Phase 2 should have user_weight=0.02"

    # Test Phase 3 weights (no context exposure)
    print("\nPhase 3: user_weight=0.0")
    phase3_config = {
        "masking": {
            "negotiation": {
                "user_weight": 0.0,
                "assistant_weight": 1.0,
                "system_weight": 0.0
            }
        }
    }
    collator.set_phase("phase3", phase3_config)
    result3 = collator.mask_example(example.copy())
    weights3 = torch.unique(result3["loss_weights"])
    print(f"Weights: {weights3.tolist()}")
    assert 0.02 not in weights3, "Phase 3 should not have user_weight=0.02"

    print("\n✓ Dynamic curriculum weights PASSED")
    return True


def main():
    print("\n" + "="*80)
    print("CURRICULUM MASKING TEST SUITE")
    print("="*80)

    tests = [
        test_next_utterance_masking,
        test_dynamic_curriculum_weights
    ]

    results = []
    for test_fn in tests:
        try:
            result = test_fn()
            results.append((test_fn.__name__, result))
        except Exception as e:
            print(f"\n✗ {test_fn.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_fn.__name__, False))

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed")
        exit(1)


if __name__ == "__main__":
    main()
