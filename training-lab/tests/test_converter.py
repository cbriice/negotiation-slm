#!/usr/bin/env python3
"""Test script to verify converter output format."""

import json
import sys
from pathlib import Path

# Add the training module to path
sys.path.insert(0, str(Path(__file__).parent))

# Import only what we need, avoiding datasets
import importlib.util
spec = importlib.util.spec_from_file_location(
    "messages_to_sft",
    "training/converters/messages_to_sft.py"
)
module = importlib.util.module_from_spec(spec)

# Patch out the datasets import
sys.modules['datasets'] = type(sys)('datasets')
sys.modules['datasets'].Dataset = None

spec.loader.exec_module(module)
MessagesToSFTConverter = module.MessagesToSFTConverter

def test_conversion():
    """Test the converter with the example packet."""

    # Read the example packet
    example_file = Path("schema/new/new_sft_packet_example.jsonl")
    with open(example_file, 'r') as f:
        packet = json.loads(f.readline())

    # Initialize converter
    converter = MessagesToSFTConverter(mask_non_assistant=True)

    # Convert the packet
    converted = converter.convert_packet(packet)

    # Print the formatted text
    print("=" * 60)
    print("CONVERTED OUTPUT:")
    print("=" * 60)
    print(converted['text'])
    print("=" * 60)
    print("\nMETADATA:")
    print(json.dumps(converted['meta'], indent=2))

    # Check masking regions if present
    if 'masking_regions' in converted:
        print("\n" + "=" * 60)
        print("MASKING REGIONS (what gets -100 loss mask):")
        print("=" * 60)

        text = converted['text']
        for i, region in enumerate(converted['masking_regions']):
            start = region['start']
            end = region['end']
            snippet = text[start:min(start+50, end)].replace('\n', '\\n')
            if end - start > 50:
                snippet += '...'

            print(f"\nRegion {i+1}:")
            print(f"  Role: {region['role']}")
            print(f"  Masked: {region['mask']}")
            print(f"  Position: {start}-{end}")
            print(f"  Content preview: {snippet}")

        # Verify what's masked vs not masked
        print("\n" + "=" * 60)
        print("MASKING SUMMARY:")
        print("=" * 60)

        roles_masked = [r['role'] for r in converted['masking_regions'] if r['mask']]
        roles_unmasked = [r['role'] for r in converted['masking_regions'] if not r['mask']]

        print(f"Masked roles (get -100 loss mask): {set(roles_masked)}")
        print(f"Unmasked roles (used for training): {set(roles_unmasked)}")

        # Count turns
        system_masked = sum(1 for r in converted['masking_regions'] if r['role'] == 'system' and r['mask'])
        user_masked = sum(1 for r in converted['masking_regions'] if r['role'] == 'user' and r['mask'])
        assistant_unmasked = sum(1 for r in converted['masking_regions'] if r['role'] == 'assistant' and not r['mask'])

        print(f"\nSystem regions masked: {system_masked}")
        print(f"User turns masked: {user_masked}")
        print(f"Assistant turns unmasked (for training): {assistant_unmasked}")

    return converted

if __name__ == "__main__":
    test_conversion()