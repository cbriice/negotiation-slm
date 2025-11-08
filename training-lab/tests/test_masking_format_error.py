#!/usr/bin/env python3
"""
Test script to reproduce the masking_regions format error.
This should fail with the current code and pass after the fix.
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
    # This should fail with current code
    dataset = pipeline.load_ndjson_dataset(temp_file)
    print(f"Loaded dataset with {len(dataset)} examples")

    # This is where it fails
    processed = pipeline.preprocess_dataset(dataset)
    print("SUCCESS: Dataset processed without error!")

except ValueError as e:
    print(f"ERROR (expected): {e}")

# Clean up
Path(temp_file).unlink()
print("Test complete")