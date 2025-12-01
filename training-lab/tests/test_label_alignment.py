#!/usr/bin/env python3
"""
Test script to diagnose label alignment issue causing high loss.
"""

import torch
from transformers import AutoTokenizer
from _legacy.utils.masking_regions_collator import DataCollatorForMaskingRegions

def test_label_alignment():
    """Test that labels are properly aligned for autoregressive training."""

    print("=" * 60)
    print("Testing Label Alignment for Autoregressive Training")
    print("=" * 60)

    # Initialize tokenizer with left padding
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Initialize collator
    collator = DataCollatorForMaskingRegions(
        tokenizer=tokenizer,
        max_length=512,
        ignore_index=-100
    )

    # Simple test example
    test_example = {
        "text": "The capital of France is Paris.",
        "masking_regions": [
            {"start": 0, "end": 25, "mask": False, "role": "text"},  # Train on everything except last word
            {"start": 25, "end": 31, "mask": False, "role": "text"}   # "Paris."
        ]
    }

    print("\nTest text:", test_example['text'])

    # Process through collator
    batch = collator([test_example])

    input_ids = batch['input_ids'][0]
    labels = batch['labels'][0]

    print("\n" + "=" * 60)
    print("Label Alignment Check:")
    print("=" * 60)
    print("For autoregressive LM, label[i] should equal input_ids[i+1]")
    print("(We predict the NEXT token)")
    print("-" * 60)

    input_ids_list = input_ids.tolist()
    labels_list = labels.tolist()

    print(f"\n{'Pos':>4} | {'Input Token':>20} | {'Input ID':>7} | {'Label':>7} | {'Expected':>7} | {'Match?':>7}")
    print("-" * 80)

    alignment_errors = 0
    for i in range(len(input_ids_list)):
        token_text = tokenizer.decode([input_ids_list[i]]).replace('\n', '\\n')

        # For autoregressive, label should be next token (or -100 if masked)
        if i < len(input_ids_list) - 1:
            expected_label = input_ids_list[i + 1] if labels_list[i] != -100 else -100
        else:
            expected_label = tokenizer.eos_token_id if labels_list[i] != -100 else -100

        if labels_list[i] != -100:
            match = "✓" if labels_list[i] == expected_label else "✗"
            if labels_list[i] != expected_label:
                alignment_errors += 1
        else:
            match = "MASKED"
            expected_label = -100

        print(f"{i:4d} | {token_text:>20} | {input_ids_list[i]:7d} | {labels_list[i]:7d} | {expected_label:7d} | {match:>7}")

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    if alignment_errors == 0:
        print("✓ Labels are correctly aligned for autoregressive training!")
    else:
        print(f"❌ CRITICAL: {alignment_errors} label alignment errors detected!")
        print("   This explains the high loss - the model is trying to predict wrong tokens!")

    # Check if labels are just a copy of input_ids
    non_masked_positions = [i for i, label in enumerate(labels_list) if label != -100]
    if non_masked_positions:
        labels_match_inputs = all(
            labels_list[i] == input_ids_list[i]
            for i in non_masked_positions
        )
        if labels_match_inputs:
            print("\n❌ CRITICAL ERROR: Labels are copies of input_ids instead of being shifted!")
            print("   For autoregressive training, labels should be input_ids shifted left by 1.")
            print("   This is why the loss is very high - the model can't predict itself!")

    return alignment_errors

if __name__ == "__main__":
    errors = test_label_alignment()