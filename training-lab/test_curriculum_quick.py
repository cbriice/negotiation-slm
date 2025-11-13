#!/usr/bin/env python3
"""
Quick test script to validate curriculum training setup.
Tests that weighted masking is working correctly.
"""

import torch
from transformers import AutoTokenizer
from training.curriculum.masking_collator import CurriculumMaskingCollator
import curriculum_parameters as cp


def test_weighted_masking():
    """Test that weighted masking produces correct loss_weights."""

    print("\n" + "="*80)
    print("TESTING WEIGHTED MASKING SETUP")
    print("="*80)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create collator
    collator = CurriculumMaskingCollator(tokenizer, "test_phase")

    # Test conversation
    test_text = """System: You are a negotiator.

User: I want to buy this car for $10,000.
Assistant: I can offer it for $12,000.
User: That's too high. How about $11,000?
Assistant: Deal! $11,000 works for me."""

    # Tokenize
    input_ids = tokenizer.encode(test_text, add_special_tokens=True)

    # Test Phase 1 config (user_weight=0.05)
    print("\nPhase 1 Config (user_weight=0.05):")
    phase1_config = cp.PHASES[0]
    collator.set_phase("phase1", phase1_config)

    example = {
        "input_ids": input_ids,
        "text": test_text,
        "source": "negotiation"
    }

    collator.mask_example(example)

    if "loss_weights" in example:
        weights = example["loss_weights"]
        print(f"✓ Weighted masking activated")
        print(f"  Total tokens: {len(weights)}")
        print(f"  Tokens with weight 0.0: {(weights == 0.0).sum().item()}")
        print(f"  Tokens with weight 0.05: {(weights == 0.05).sum().item()}")
        print(f"  Tokens with weight 1.0: {(weights == 1.0).sum().item()}")
        print(f"  Average weight: {weights.mean().item():.3f}")
    else:
        print("✗ WARNING: No loss_weights in example - weighted masking not working!")
        return False

    # Test Phase 3 config (user_weight=0.0)
    print("\nPhase 3 Config (user_weight=0.0):")
    phase3_config = cp.PHASES[2]
    collator.set_phase("phase3", phase3_config)

    example2 = {
        "input_ids": input_ids,
        "text": test_text,
        "source": "negotiation"
    }

    collator.mask_example(example2)

    if "loss_weights" in example2:
        weights = example2["loss_weights"]
        print(f"✓ Weighted masking activated")
        print(f"  Total tokens: {len(weights)}")
        print(f"  Tokens with weight 0.0: {(weights == 0.0).sum().item()}")
        print(f"  Tokens with weight 1.0: {(weights == 1.0).sum().item()}")
        print(f"  Average weight: {weights.mean().item():.3f}")

        # Verify no user turns are weighted in phase 3
        if (weights == 0.05).sum().item() == 0:
            print("✓ Phase 3 correctly has no user turn weights")
        else:
            print("✗ WARNING: Phase 3 still has user turn weights!")
            return False
    else:
        print("✗ WARNING: No loss_weights in example!")
        return False

    print("\n" + "="*80)
    print("✓ WEIGHTED MASKING TEST PASSED")
    print("="*80 + "\n")
    return True


def print_curriculum_config():
    """Print the current curriculum configuration."""
    print("\n" + "="*80)
    print("CURRENT CURRICULUM CONFIGURATION")
    print("="*80)
    print(f"Total training steps: {cp.MAX_STEPS}")
    print(f"Number of phases: {len(cp.PHASES)}\n")

    for i, phase in enumerate(cp.PHASES, 1):
        print(f"Phase {i}: {phase['name']}")
        print(f"  Steps: {phase['step_start']} → {phase['step_end']}")
        print(f"  Duration: {phase['step_end'] - phase['step_start']} steps")
        print(f"  Data mix: Conv {phase['data_ratio']['conversational']:.0%} / Neg {phase['data_ratio']['negotiation']:.0%}")

        # Print masking weights for each data source
        for source, weights in phase['masking'].items():
            if isinstance(weights, dict):
                user_w = weights.get('user_weight', 0.0)
                asst_w = weights.get('assistant_weight', 1.0)
                print(f"  {source.capitalize()} masking: User {user_w:.3f} / Assistant {asst_w:.1f}")
        print()

    print("="*80 + "\n")


if __name__ == "__main__":
    # Print configuration
    print_curriculum_config()

    # Test weighted masking
    success = test_weighted_masking()

    if success:
        print("\n✓ All tests passed! Ready to train.")
        print("\nQuick start commands:")
        print("  # 500-step diagnostic:")
        print("  python train_curriculum.py --max-steps 500 --output-dir models/diagnostic_500")
        print("\n  # 2000-step validation:")
        print("  python train_curriculum.py --max-steps 2000 --output-dir models/validation_2k")
        print("\n  # Full training (uses MAX_STEPS from curriculum_parameters.py):")
        print("  python train_curriculum.py --output-dir models/curriculum_weighted")
        print()
    else:
        print("\n✗ Tests failed. Check configuration.")
        exit(1)
