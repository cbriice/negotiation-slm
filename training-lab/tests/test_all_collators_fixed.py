#!/usr/bin/env python3
"""
Test script to verify all collators have correct label alignment after fixes.
"""

import torch
from transformers import AutoTokenizer
from _legacy.utils.masking_regions_collator import DataCollatorForMaskingRegions
from _legacy.utils.json_data_collator import DataCollatorForJSON
from _legacy.utils.weighted_loss_collator import WeightedLossCollator

def test_collator_alignment(collator_name, collator, test_data):
    """Test that a collator has correct label alignment."""

    print("\n" + "=" * 60)
    print(f"Testing {collator_name}")
    print("=" * 60)

    # Process through collator
    batch = collator([test_data])

    input_ids = batch['input_ids'][0]
    labels = batch['labels'][0]

    input_ids_list = input_ids.tolist()
    labels_list = labels.tolist()

    # Check alignment
    alignment_errors = 0
    for i in range(len(labels_list)):
        if labels_list[i] != -100:  # Only check non-masked positions
            # Label should be next token
            if i < len(input_ids_list) - 1:
                expected = input_ids_list[i + 1]
            else:
                expected = collator.tokenizer.eos_token_id

            if labels_list[i] != expected:
                alignment_errors += 1
                if alignment_errors <= 3:  # Show first 3 errors
                    token = collator.tokenizer.decode([input_ids_list[i]])
                    label_token = collator.tokenizer.decode([labels_list[i]])
                    expected_token = collator.tokenizer.decode([expected])
                    print(f"  Error at pos {i}: predicting '{label_token}' from '{token}', expected '{expected_token}'")

    if alignment_errors == 0:
        print(f"✓ {collator_name}: Labels correctly aligned!")
    else:
        print(f"✗ {collator_name}: {alignment_errors} alignment errors found!")

    # Check that some positions are being trained
    trained_positions = sum(1 for label in labels_list if label != -100)
    print(f"  Training on {trained_positions}/{len(labels_list)} positions ({trained_positions/len(labels_list)*100:.1f}%)")

    return alignment_errors == 0


def main():
    """Test all collators."""

    print("=" * 60)
    print("Testing All Collators After Fix")
    print("=" * 60)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    all_passed = True

    # Test 1: DataCollatorForMaskingRegions
    collator1 = DataCollatorForMaskingRegions(
        tokenizer=tokenizer,
        max_length=512,
        ignore_index=-100
    )

    test_data1 = {
        "text": "User: What is 2+2?\nAssistant: 2+2 equals 4.",
        "masking_regions": [
            {"start": 0, "end": 19, "mask": True, "role": "user"},
            {"start": 19, "end": 44, "mask": False, "role": "assistant"}
        ]
    }

    passed = test_collator_alignment("DataCollatorForMaskingRegions", collator1, test_data1)
    all_passed = all_passed and passed

    # Test 2: DataCollatorForJSON
    collator2 = DataCollatorForJSON(
        tokenizer=tokenizer,
        max_length=512,
        ignore_index=-100
    )

    test_data2 = {
        "input": "What is the capital of France?",
        "output": "The capital of France is Paris."
    }

    passed = test_collator_alignment("DataCollatorForJSON", collator2, test_data2)
    all_passed = all_passed and passed

    # Test 3: WeightedLossCollator
    collator3 = WeightedLossCollator(
        tokenizer=tokenizer,
        max_length=512,
        ignore_index=-100,
        prompt_weight=0.15
    )

    test_data3 = {
        "system": "You are a helpful assistant.",
        "prompt": "What is 2+2?",
        "response": "2+2 equals 4."
    }

    passed = test_collator_alignment("WeightedLossCollator", collator3, test_data3)
    all_passed = all_passed and passed

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_passed:
        print("✅ ALL COLLATORS PASSED - Label alignment is fixed!")
        print("   The high loss issue should now be resolved.")
    else:
        print("❌ SOME COLLATORS FAILED - Check the errors above.")

    print("\nNext steps:")
    print("1. Restart your training with the fixed collators")
    print("2. Loss should now be reasonable (typically starts around 2-3)")
    print("3. Monitor that loss decreases during training")


if __name__ == "__main__":
    main()