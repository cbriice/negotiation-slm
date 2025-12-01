#!/usr/bin/env python3
"""
Test script to verify masking behavior and diagnose high loss issue.
"""

import torch
from transformers import AutoTokenizer
from _legacy.utils.masking_regions_collator import DataCollatorForMaskingRegions

def test_masking():
    """Test that masking is applied correctly to the right parts."""

    print("=" * 60)
    print("Testing Masking Behavior for High Loss Diagnosis")
    print("=" * 60)

    # Initialize tokenizer with left padding
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For Flash Attention

    # Initialize collator
    collator = DataCollatorForMaskingRegions(
        tokenizer=tokenizer,
        max_length=512,
        ignore_index=-100
    )

    # Create test example with clear structure
    test_example = {
        "text": "System: You are a helpful assistant.\n\nUser: What is 2+2?\n\nAssistant: 2+2 equals 4.",
        "masking_regions": [
            {"start": 0, "end": 38, "mask": True, "role": "system"},  # "System: You are a helpful assistant.\n\n"
            {"start": 38, "end": 56, "mask": True, "role": "user"},    # "User: What is 2+2?\n\n"
            {"start": 56, "end": 80, "mask": False, "role": "assistant"} # "Assistant: 2+2 equals 4."
        ]
    }

    print("\nTest example:")
    print(f"Text: {test_example['text']}")
    print(f"\nMasking regions:")
    for region in test_example['masking_regions']:
        text_slice = test_example['text'][region['start']:region['end']]
        print(f"  {region['role']}: mask={region['mask']} -> '{text_slice}'")

    # Process through collator
    batch = collator([test_example])

    print("\n" + "=" * 60)
    print("Tokenization Results:")
    print("=" * 60)

    # Decode tokens to see what we have
    input_ids = batch['input_ids'][0]
    labels = batch['labels'][0]

    # Convert to list for easier inspection
    input_ids_list = input_ids.tolist()
    labels_list = labels.tolist()

    # Decode each token
    print("\nToken-by-token analysis:")
    print("-" * 60)

    non_masked_count = 0
    masked_count = 0

    for i, (input_id, label) in enumerate(zip(input_ids_list, labels_list)):
        token_text = tokenizer.decode([input_id])
        if label == -100:
            status = "MASKED"
            masked_count += 1
        else:
            status = f"TRAINING (label={label})"
            non_masked_count += 1

        # Only show first 30 and last 10 tokens for brevity
        if i < 30 or i >= len(input_ids_list) - 10:
            print(f"Token {i:3d}: {token_text:20s} | {status}")
        elif i == 30:
            print(f"... {len(input_ids_list) - 40} tokens omitted ...")

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Total tokens: {len(input_ids_list)}")
    print(f"Masked tokens (loss=-100): {masked_count}")
    print(f"Training tokens (loss computed): {non_masked_count}")
    print(f"Percentage being trained on: {non_masked_count / len(input_ids_list) * 100:.2f}%")

    # Check if assistant response is actually being trained on
    assistant_text = "2+2 equals 4"
    assistant_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
    print(f"\nAssistant response tokens: {assistant_tokens}")
    print(f"Decoded: {tokenizer.decode(assistant_tokens)}")

    # Check for common issues
    print("\n" + "=" * 60)
    print("Diagnostic Checks:")
    print("=" * 60)

    if non_masked_count == 0:
        print("❌ WARNING: All tokens are masked! No loss will be computed.")
    elif non_masked_count < 10:
        print("⚠️  WARNING: Very few tokens are unmasked. Loss may be unstable.")
    else:
        print("✓ Sufficient tokens are unmasked for training.")

    if masked_count == 0:
        print("❌ WARNING: No tokens are masked! Training on entire sequence.")
    else:
        print(f"✓ {masked_count} tokens are properly masked.")

    # Check padding
    pad_token_id = tokenizer.pad_token_id
    padding_count = sum(1 for tid in input_ids_list if tid == pad_token_id)
    if padding_count > 0:
        print(f"✓ Found {padding_count} padding tokens (left-padded for Flash Attention).")
        # Check if padding is at the beginning (left padding)
        if input_ids_list[0] == pad_token_id:
            print("✓ Padding is correctly at the beginning (left padding).")
        else:
            print("❌ WARNING: Padding is not at the beginning despite left padding setting!")

    return batch

if __name__ == "__main__":
    test_masking()