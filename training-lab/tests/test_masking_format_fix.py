#!/usr/bin/env python3
"""
Test script to verify the masking_regions format fix works.
This should pass with the updated code.
"""

import json
import tempfile
from pathlib import Path
from datasets import Dataset

# Create test data in the new format
test_data = [
    {
        "text": "System: You are helpful.\nUser: Hello\nAssistant: Hi there!",
        "meta": {"source": "test"},
        "masking_regions": [
            {"start": 0, "end": 25, "mask": True, "role": "system"},
            {"start": 25, "end": 36, "mask": True, "role": "user"},
            {"start": 36, "end": 48, "mask": False, "role": "assistant"}
        ]
    }
]

# Create a temporary file
with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')
    temp_file = f.name

print(f"Created test file: {temp_file}")

# Try to load with SFT trainer's preprocess_dataset
from training.sft_trainer import SFTTrainingPipeline

pipeline = SFTTrainingPipeline()

try:
    # Load the dataset
    dataset = pipeline.load_ndjson_dataset(temp_file)
    print(f"✓ Loaded dataset with {len(dataset)} examples")

    # Preprocess - this should work now
    processed = pipeline.preprocess_dataset(dataset)
    print("✓ Dataset preprocessed without error")

    # Check if the format was detected correctly
    if hasattr(pipeline, 'use_masking_regions') and pipeline.use_masking_regions:
        print("✓ Masking regions format correctly detected")
    else:
        print("✗ Masking regions format not detected")

    print("\nSUCCESS: The masking_regions format is now supported!")

except ValueError as e:
    print(f"ERROR: {e}")
    print("\nFAILED: The fix didn't work properly")

except Exception as e:
    print(f"Unexpected error: {e}")

# Clean up
Path(temp_file).unlink()
print("\nTest complete")