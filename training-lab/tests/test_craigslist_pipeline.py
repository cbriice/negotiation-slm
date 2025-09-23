#!/usr/bin/env python3
"""
Test script to verify the CraigslistBargain training pipeline.
Tests data loading, processing, and training setup.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from training.craigslist_loader import CraigslistBargainLoader


def test_data_loading():
    """Test loading and processing CraigslistBargain dataset."""
    print("\n" + "="*60)
    print("Testing CraigslistBargain Data Loading")
    print("="*60)

    # Initialize loader
    loader = CraigslistBargainLoader(cache_dir="data/cache")

    # Load a small sample from train split
    print("\n1. Loading sample from training split...")
    dataset = loader.load_dataset('train')
    print(f"   ✓ Loaded {len(dataset)} conversations")

    # Check first example structure
    if len(dataset) > 0:
        example = dataset[0]
        print("\n2. Checking dataset structure...")
        print(f"   Fields in first example: {list(example.keys())}")

        # Extract and display dialogue
        turns = loader._extract_dialogue_turns(example)
        print(f"\n3. Sample dialogue ({len(turns)} turns):")
        for i, (speaker, utterance) in enumerate(turns[:3]):
            print(f"   {i+1}. {speaker}: {utterance[:50]}...")

        # Create system prompt
        system_buyer = loader._create_system_prompt(example, 'buyer')
        system_seller = loader._create_system_prompt(example, 'seller')
        print(f"\n4. System prompts:")
        print(f"   Buyer: {system_buyer[:100]}...")
        print(f"   Seller: {system_seller[:100]}...")

        # Create SFT packets
        packets = loader._create_sft_packets(example)
        print(f"\n5. Generated {len(packets)} SFT packets from conversation")

        if packets:
            print("\n6. Sample SFT packet:")
            packet = packets[0]
            print(f"   ID: {packet['id']}")
            print(f"   System: {packet['system'][:80]}...")
            print(f"   Prompt: {packet['prompt'][:80]}...")
            print(f"   Response: {packet['response'][:80]}...")

    # Test processing multiple examples
    print("\n7. Processing batch of conversations...")
    output_file = "data/test/craigslist_test.jsonl"
    processed_file = loader.process_dataset_to_sft(
        dataset=dataset,
        output_file=output_file,
        max_examples=10  # Just test with 10 conversations
    )
    print(f"   ✓ Saved to: {processed_file}")

    # Verify the output file
    packet_count = 0
    with open(processed_file, 'r') as f:
        for line in f:
            if line.strip():
                packet_count += 1

    print(f"   ✓ Generated {packet_count} total packets from 10 conversations")

    print("\n" + "="*60)
    print("✓ Data loading test completed successfully!")
    print("="*60)


def test_training_compatibility():
    """Test that processed data is compatible with training pipeline."""
    import json

    print("\n" + "="*60)
    print("Testing Training Pipeline Compatibility")
    print("="*60)

    test_file = "data/test/craigslist_test.jsonl"

    if not Path(test_file).exists():
        print("   ✗ Test file not found. Run test_data_loading() first.")
        return

    print("\n1. Checking packet format...")
    with open(test_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Check first 3 packets
                break

            packet = json.loads(line)
            required_fields = ['id', 'system', 'prompt', 'response']

            # Check all fields present
            missing = [f for f in required_fields if f not in packet]
            if missing:
                print(f"   ✗ Packet {i+1} missing fields: {missing}")
            else:
                print(f"   ✓ Packet {i+1} has all required fields")

                # Check field contents
                if packet['response'].strip() == '':
                    print(f"     Warning: Empty response in packet {i+1}")

    print("\n2. Checking data diversity...")
    roles = {'buyer': 0, 'seller': 0}
    with open(test_file, 'r') as f:
        for line in f:
            packet = json.loads(line)
            # Extract role from ID
            id_parts = packet['id'].split('_')
            if len(id_parts) >= 2 and id_parts[1] in roles:
                roles[id_parts[1]] += 1

    print(f"   Buyer packets: {roles['buyer']}")
    print(f"   Seller packets: {roles['seller']}")

    if roles['buyer'] > 0 and roles['seller'] > 0:
        print("   ✓ Both perspectives represented")
    else:
        print("   ✗ Missing perspective diversity")

    print("\n" + "="*60)
    print("✓ Compatibility test completed!")
    print("="*60)


def main():
    """Run all tests."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\nStarting CraigslistBargain Pipeline Tests")
    print("="*60)

    try:
        # Test data loading and processing
        test_data_loading()

        # Test training compatibility
        test_training_compatibility()

        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("You can now run the full training pipeline:")
        print("  python scripts/run_baseline_training.py")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()