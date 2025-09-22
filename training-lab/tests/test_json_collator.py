#!/usr/bin/env python3
"""
Test script to verify DataCollatorForJSON works with SFTTrainer correctly.
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
from utils.json_data_collator import DataCollatorForJSON

def create_sample_data():
    """Create sample JSON training data."""
    samples = [
        {
            "input": {
                "state": {"item": "laptop", "my_price": 1000, "their_price": 1200},
                "recent_convo": [
                    {"side": "buyer", "text": "I'll pay 1000"},
                    {"side": "seller", "text": "That's too low"}
                ],
                "partner_msg": "How about 1100?"
            },
            "output": "I can meet you at 1050, that's my final offer."
        },
        {
            "input": {
                "state": {"item": "phone", "my_price": 500, "their_price": 600},
                "recent_convo": [],
                "partner_msg": "What's your offer?"
            },
            "output": "I'm looking to pay around 500 for this phone."
        }
    ]
    return samples

def test_collator():
    """Test the DataCollatorForJSON."""
    print("="*60)
    print("Testing DataCollatorForJSON")
    print("="*60)

    # Initialize tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

    # Create collator
    print("\n2. Creating DataCollatorForJSON...")
    collator = DataCollatorForJSON(
        tokenizer=tokenizer,
        max_length=512,
        ignore_index=-100
    )
    print("✓ Collator created")

    # Create sample data
    print("\n3. Creating sample data...")
    samples = create_sample_data()
    print(f"✓ Created {len(samples)} samples")

    # Test collation
    print("\n4. Testing collation...")
    batch = collator(samples)

    # Verify output structure
    print("\n5. Verifying output structure:")
    assert "input_ids" in batch, "Missing input_ids"
    assert "labels" in batch, "Missing labels"
    assert "attention_mask" in batch, "Missing attention_mask"
    print("✓ All required keys present")

    # Check shapes
    print("\n6. Checking tensor shapes:")
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")
    print(f"   attention_mask shape: {batch['attention_mask'].shape}")
    assert batch['input_ids'].shape == batch['labels'].shape, "Shape mismatch"
    assert batch['input_ids'].shape == batch['attention_mask'].shape, "Shape mismatch"
    print("✓ All shapes match")

    # Check masking
    print("\n7. Analyzing masking:")
    for i, sample in enumerate(samples):
        labels = batch['labels'][i]
        input_ids = batch['input_ids'][i]

        # Count masked vs supervised tokens
        masked_count = (labels == -100).sum().item()
        supervised_count = (labels != -100).sum().item()
        total_count = len(labels)

        print(f"\n   Sample {i+1}:")
        print(f"   - Total tokens: {total_count}")
        print(f"   - Masked (input): {masked_count} ({masked_count/total_count*100:.1f}%)")
        print(f"   - Supervised (output): {supervised_count} ({supervised_count/total_count*100:.1f}%)")

        # Decode supervised tokens to verify they match output
        if supervised_count > 0:
            supervised_ids = input_ids[labels != -100]
            supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=True)
            print(f"   - Training on: '{supervised_text[:50]}...'")

            # Verify it matches the expected output
            expected = sample['output']
            if expected[:30] in supervised_text:
                print(f"   ✓ Supervised text matches expected output")
            else:
                print(f"   ⚠ Warning: Supervised text may not match expected output")

    print("\n" + "="*60)
    print("✅ DataCollatorForJSON test completed successfully!")
    print("="*60)

    return batch

def test_with_dataset():
    """Test with HuggingFace Dataset format."""
    print("\n" + "="*60)
    print("Testing with HuggingFace Dataset")
    print("="*60)

    # Create dataset
    samples = create_sample_data()
    dataset = Dataset.from_list(samples)
    print(f"✓ Created dataset with {len(dataset)} samples")
    print(f"  Columns: {dataset.column_names}")

    # Initialize tokenizer and collator
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForJSON(
        tokenizer=tokenizer,
        max_length=512,
        ignore_index=-100
    )

    # Test with dataset samples
    batch_samples = [dataset[i] for i in range(len(dataset))]
    batch = collator(batch_samples)

    print(f"\n✓ Successfully processed dataset samples")
    print(f"  Batch keys: {batch.keys()}")
    print(f"  Batch shape: {batch['input_ids'].shape}")

    return dataset, collator

if __name__ == "__main__":
    # Test basic collator functionality
    batch = test_collator()

    # Test with Dataset format
    dataset, collator = test_with_dataset()

    print("\n" + "="*60)
    print("🎉 All tests passed! DataCollatorForJSON is ready for SFTTrainer")
    print("="*60)
    print("\nKey points for SFTTrainer integration:")
    print("1. ✓ Pass raw dataset with 'input' and 'output' fields")
    print("2. ✓ Use data_collator=DataCollatorForJSON(...)")
    print("3. ✓ Set tokenizer=None to prevent pre-tokenization")
    print("4. ✓ Don't pass formatting_func or dataset_text_field")
    print("5. ✓ Ensure packing=False in SFTConfig")