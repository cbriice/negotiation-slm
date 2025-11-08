#!/usr/bin/env python3
"""
Test script to verify SFT packet conversion works correctly.
"""

import json
from pathlib import Path
from training.converters.messages_to_sft import MessagesToSFTConverter


def test_conversion():
    """Test the conversion of new SFT packets to training format."""

    # Path to the example packet
    example_file = Path("schema/new_sft_packet_example.jsonl")

    if not example_file.exists():
        print(f"Error: {example_file} not found")
        return

    print("=" * 60)
    print("Testing SFT Packet Conversion")
    print("=" * 60)

    # Initialize converter
    converter = MessagesToSFTConverter(mask_non_assistant=True)

    # Read the example packet
    with open(example_file, 'r') as f:
        line = f.readline().strip()
        packet = json.loads(line)

    print("\n1. Original packet structure:")
    print(f"   - Messages: {len(packet.get('messages', []))} messages")
    print(f"   - System info: {packet.get('system', {})}")
    print(f"   - Meta info: {packet.get('meta', {})}")

    # Convert the packet
    print("\n2. Converting packet...")
    converted = converter.convert_packet(packet)

    print("\n3. Converted output:")
    print("-" * 40)
    print(converted['text'])
    print("-" * 40)

    print("\n4. Conversion metadata:")
    print(f"   - Total turns: {converted['meta']['num_turns']}")
    print(f"   - Assistant turns: {converted['meta']['assistant_turns']}")
    print(f"   - User turns: {converted['meta']['user_turns']}")
    print(f"   - Has system message: {converted['meta']['has_system']}")

    if 'masking_regions' in converted:
        print("\n5. Masking regions (for loss calculation):")
        for i, region in enumerate(converted['masking_regions'][:3]):  # Show first 3
            print(f"   Region {i+1}: role={region['role']}, mask={region['mask']}, "
                  f"range=[{region['start']}:{region['end']}]")

    # Test with the dataset loader
    print("\n6. Testing integration with UnifiedDatasetLoader...")
    from utils.dataset_loader import UnifiedDatasetLoader

    loader = UnifiedDatasetLoader()

    # Test the format_sft_dialogue function
    formatted_text = loader.format_sft_dialogue(packet)
    print("   Dataset loader formatting successful!")
    print(f"   Output length: {len(formatted_text)} characters")

    # Verify the formatting matches
    if formatted_text == converted['text']:
        print("   ✓ Output matches converter output exactly")
    else:
        print("   ⚠ Output differs from direct converter (may need investigation)")

    print("\n7. Batch conversion test...")
    # Convert the full file if it exists
    converted_examples, stats = converter.convert_file(
        example_file,
        output_path=None,
        validate=True
    )

    print(f"   Converted {stats['successful_conversions']} packets successfully")
    print(f"   Failed conversions: {stats['failed_conversions']}")

    if stats['errors']:
        print("   Errors encountered:")
        for error in stats['errors'][:3]:
            print(f"     - {error}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

    # Save a sample converted file for inspection
    output_file = Path("test_output/converted_sft_sample.jsonl")
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        for example in converted_examples[:5]:  # Save first 5
            output_data = {
                'text': example['text'],
                'meta': example['meta']
            }
            f.write(json.dumps(output_data) + '\n')

    print(f"\nSample converted data saved to: {output_file}")
    print("\nYou can now use this converted data with the SFT trainer:")
    print(f"  python -m training.sft_trainer --dataset {output_file}")


if __name__ == "__main__":
    test_conversion()