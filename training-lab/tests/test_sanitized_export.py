#!/usr/bin/env python3
"""
Test script to verify sanitized Deal or No Deal data extraction.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.converters.builtin_export import BuiltinTaskExporter


def test_sanitization():
    """Test that sanitization removes prompts and metadata."""
    exporter = BuiltinTaskExporter()

    # Test cases with contaminated data
    test_cases = [
        # Case 1: Data with prompts
        {
            'input': "## Buyer\n\nYou are a buyer negotiating for Laptop.\nYour target price is $1000.\n\nMake a negotiation offer.",
            'expected': "",  # Should be empty after sanitization
        },
        # Case 2: Clean dialogue
        {
            'input': "I'd like to offer $850 for the laptop. It's a fair price considering the market.",
            'expected': "I'd like to offer $850 for the laptop. It's a fair price considering the market.",
        },
        # Case 3: Mixed content
        {
            'input': "[PROPOSE] I can offer you $900. ### What is the price?\nThat's my best offer.",
            'expected': "I can offer you $900. \nThat's my best offer.",
        },
        # Case 4: Selection markers
        {
            'input': "deal <eos> <selection>",
            'expected': "deal",
        }
    ]

    print("=" * 80)
    print("TESTING SANITIZATION")
    print("=" * 80)

    for i, case in enumerate(test_cases, 1):
        result = exporter._sanitize_dialogue_text(case['input'])
        passed = result.strip() == case['expected'].strip()

        print(f"\nTest Case {i}: {'✓ PASSED' if passed else '✗ FAILED'}")
        print(f"Input: {case['input'][:50]}...")
        print(f"Expected: {case['expected'][:50]}...")
        print(f"Got: {result[:50]}...")

        if not passed:
            print(f"FULL OUTPUT: {result}")


def test_validation():
    """Test validation of negotiation text."""
    exporter = BuiltinTaskExporter()

    test_cases = [
        # Valid negotiation text
        ("I can offer you $850 for the item", True),
        ("That's a deal!", True),
        ("How about we split the difference?", True),
        ("I accept your offer", True),

        # Invalid - prompts/metadata
        ("## Negotiation", False),
        ("You are a buyer", False),
        ("Your target price is $900", False),
        ("Make a negotiation offer", False),
        ("Subject: Offer for Laptop", False),
        ("1. First point", False),
        ("", False),
        ("xyz", False),  # Too short
    ]

    print("\n" + "=" * 80)
    print("TESTING VALIDATION")
    print("=" * 80)

    for text, expected in test_cases:
        result = exporter._is_valid_negotiation_text(text)
        passed = result == expected

        status = "✓" if passed else "✗"
        print(f"{status} '{text[:40]}...' -> Expected: {expected}, Got: {result}")


def test_export_samples():
    """Export a few Deal or No Deal samples to inspect."""
    print("\n" + "=" * 80)
    print("EXPORTING SAMPLE EPISODES")
    print("=" * 80)

    output_dir = "data/test_sanitized"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    exporter = BuiltinTaskExporter(output_dir=output_dir)

    # Export just a few episodes for inspection
    episodes = exporter.export_task('dealnodeal', max_episodes=5, datatype='train')

    if episodes:
        # Save episodes
        output_file = Path(output_dir) / "dealnodeal_sanitized_samples.jsonl"
        with open(output_file, 'w') as f:
            for episode in episodes:
                f.write(json.dumps(episode) + '\n')

        print(f"\nExported {len(episodes)} episodes to {output_file}")

        # Display first episode for inspection
        if episodes:
            print("\n" + "=" * 80)
            print("FIRST EPISODE SAMPLE:")
            print("=" * 80)

            episode = episodes[0]
            print(f"\nState: {json.dumps(episode.get('state', {}), indent=2)}")
            print(f"\nDialogue ({len(episode.get('dialogue', []))} turns):")

            for i, turn in enumerate(episode.get('dialogue', [])):
                print(f"\nTurn {i} ({turn.get('side', 'unknown')}):")
                if turn.get('text'):
                    print(f"  Text: {turn['text'][:100]}...")
                if turn.get('response'):
                    print(f"  Response: {turn['response'][:100]}...")
    else:
        print("No episodes exported - check if ParlAI data is available")


def check_for_contamination(episodes_file: str):
    """Check exported episodes for contamination."""
    print("\n" + "=" * 80)
    print(f"CHECKING FOR CONTAMINATION IN: {episodes_file}")
    print("=" * 80)

    if not Path(episodes_file).exists():
        print(f"File not found: {episodes_file}")
        return

    contamination_markers = [
        '##', '###',  # Markdown headers
        'You are a', 'Your target',  # Prompts
        'Make a negotiation',  # Instructions
        '[PROPOSE]', '[COMMIT]',  # Action tags (should be in metadata, not text)
        'What is the',  # Metadata questions
        'Subject:',  # Email headers
    ]

    issues = []
    total_turns = 0

    with open(episodes_file) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            episode = json.loads(line)

            for turn in episode.get('dialogue', []):
                total_turns += 1

                # Check text field
                text = turn.get('text', '')
                for marker in contamination_markers:
                    if marker in text:
                        issues.append(f"Line {line_num}, text contains: '{marker}'")

                # Check response field
                response = turn.get('response', '')
                for marker in contamination_markers:
                    if marker in response:
                        issues.append(f"Line {line_num}, response contains: '{marker}'")

    print(f"\nAnalyzed {total_turns} dialogue turns")

    if issues:
        print(f"\n⚠️  Found {len(issues)} potential contamination issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("✓ No contamination detected!")


def main():
    """Run all tests."""
    print("Testing Deal or No Deal Data Sanitization")
    print("=" * 80)

    # Run tests
    test_sanitization()
    test_validation()
    test_export_samples()

    # Check existing exported data if available
    existing_export = "data/training/builtin_tasks/dealnodeal_episodes.jsonl"
    if Path(existing_export).exists():
        check_for_contamination(existing_export)

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the exported samples in data/test_sanitized/")
    print("2. If clean, re-export full dataset with: python scripts/run_baseline_training.py")
    print("3. Check new exports for contamination before training")


if __name__ == "__main__":
    main()