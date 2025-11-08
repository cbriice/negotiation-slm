#!/usr/bin/env python3
"""
Test script to verify that the masking_regions_collator correctly handles
the output from the messages_to_sft converter.
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
from utils.masking_regions_collator import DataCollatorForMaskingRegions
import training_parameters as params


def test_masking_compatibility():
    """Test that the collator correctly applies masking from converter output."""

    print("=" * 60)
    print("Testing Masking Compatibility")
    print("=" * 60)

    # Load tokenizer
    model_name = "mistralai/Mistral-7B-v0.1"  # Default model used in sft_trainer.py
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize the collator
    collator = DataCollatorForMaskingRegions(
        tokenizer=tokenizer,
        max_length=params.MAX_SEQ_LEN,
        pad_to_multiple_of=params.PAD_TO_MULTIPLE,
        ignore_index=params.MASKING_IGNORE_INDEX
    )

    # Load a sample from the formatted dataset
    formatted_path = Path("data/training_data/DATASET-V1_formatted.jsonl")

    print(f"\nLoading samples from: {formatted_path}")

    samples = []
    with open(formatted_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Test first 3 samples
                break
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples for testing")

    # Test each sample individually
    for i, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}:")
        print(f"{'='*60}")

        # Show sample info
        text = sample['text']
        masking_regions = sample.get('masking_regions', [])

        print(f"Text length: {len(text)} chars")
        print(f"Number of masking regions: {len(masking_regions)}")

        # Show masking regions summary
        roles_masked = set()
        roles_unmasked = set()
        for region in masking_regions:
            if region['mask']:
                roles_masked.add(region['role'])
            else:
                roles_unmasked.add(region['role'])

        print(f"Roles to mask (get -100): {roles_masked}")
        print(f"Roles to train on: {roles_unmasked}")

        # Validate masking with the collator
        stats = collator.validate_masking(sample)

        print(f"\nTokenization Statistics:")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Masked tokens: {stats['total_masked']} ({stats['masking_ratio']:.1%})")
        print(f"  Unmasked tokens: {stats['total_unmasked']} ({1-stats['masking_ratio']:.1%})")

        print(f"\nTokens by role:")
        print("  Masked (will get -100 loss):")
        for role, count in stats['masked_tokens_by_role'].items():
            print(f"    {role}: {count} tokens")
        print("  Unmasked (will train on):")
        for role, count in stats['unmasked_tokens_by_role'].items():
            print(f"    {role}: {count} tokens")

        print(f"\nUnmasked text preview (what model trains on):")
        print(f"  '{stats['unmasked_text_preview']}'")

        # Verify expected behavior
        if 'system' in roles_masked and 'assistant' in roles_unmasked:
            print(f"\n✓ Masking is correct: system masked, assistant unmasked")
        else:
            print(f"\n⚠️  WARNING: Unexpected masking pattern!")

    # Test batch collation
    print(f"\n{'='*60}")
    print("Testing Batch Collation")
    print(f"{'='*60}")

    batch = collator(samples)

    print(f"Batch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")

    # Check that labels have -100 for masked regions
    for i in range(len(samples)):
        labels = batch['labels'][i]
        num_masked = (labels == params.MASKING_IGNORE_INDEX).sum().item()
        num_unmasked = (labels != params.MASKING_IGNORE_INDEX).sum().item()
        print(f"\nSample {i+1} in batch:")
        print(f"  Masked tokens (-100): {num_masked}")
        print(f"  Unmasked tokens: {num_unmasked}")

    print(f"\n{'='*60}")
    print("✓ Masking compatibility test complete!")
    print(f"{'='*60}")

    return True


if __name__ == "__main__":
    test_masking_compatibility()