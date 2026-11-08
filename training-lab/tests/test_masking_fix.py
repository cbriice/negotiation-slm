#!/usr/bin/env python3
"""
Test script to verify that only the final assistant turn is unmasked.
"""


from reformat_to_next_utterance import create_masking_regions


def test_single_assistant_turn():
    """Test conversation with single assistant turn."""
    text = """---
system: You are a helpful assistant.
---
User: Hello
Assistant: Hi there!"""

    regions = create_masking_regions(text)

    # Find assistant regions
    assistant_regions = [r for r in regions if r['role'] == 'assistant']

    print("Test 1: Single Assistant Turn")
    print(f"  Total regions: {len(regions)}")
    print(f"  Assistant regions: {len(assistant_regions)}")

    assert len(assistant_regions) == 1, "Should have exactly 1 assistant region"
    assert assistant_regions[0]['mask'] == False, "Single assistant turn should NOT be masked"

    print("  ✓ Single assistant turn correctly unmasked")
    print()


def test_multiple_assistant_turns():
    """Test conversation with multiple assistant turns (critical case)."""
    text = """---
system: You are a helpful assistant.
---
User: Hello
Assistant: Hi there!
User: How are you?
Assistant: I'm doing well, thanks for asking!
User: Great
Assistant: Is there anything I can help you with?"""

    regions = create_masking_regions(text)

    # Find assistant regions
    assistant_regions = [r for r in regions if r['role'] == 'assistant']

    print("Test 2: Multiple Assistant Turns (CRITICAL)")
    print(f"  Total regions: {len(regions)}")
    print(f"  Assistant regions: {len(assistant_regions)}")
    print()

    assert len(assistant_regions) == 3, "Should have exactly 3 assistant regions"

    # Check each assistant region
    for i, region in enumerate(assistant_regions):
        is_final = (i == len(assistant_regions) - 1)
        expected_mask = not is_final  # Final should be False, others True

        print(f"  Assistant turn {i+1}:")
        print(f"    mask={region['mask']} (expected={expected_mask})")

        assert region['mask'] == expected_mask, \
            f"Assistant turn {i+1}: Expected mask={expected_mask}, got mask={region['mask']}"

        if is_final:
            print(f"    ✓ Final assistant turn correctly UNMASKED (receives assistant_weight)")
        else:
            print(f"    ✓ Context assistant turn correctly MASKED (receives user_weight)")

    print()


def test_no_overlapping_assignments():
    """Test that regions don't overlap."""
    text = """---
system: You are a helpful assistant.
---
User: Hello
Assistant: Hi there!
User: How are you?
Assistant: I'm doing well, thanks for asking!"""

    regions = create_masking_regions(text)

    print("Test 3: No Overlapping Regions")
    print(f"  Total regions: {len(regions)}")

    # Check for overlaps
    for i in range(len(regions) - 1):
        current_end = regions[i]['end']
        next_start = regions[i+1]['start']

        assert current_end <= next_start, \
            f"Region {i} overlaps with region {i+1}: " \
            f"current_end={current_end}, next_start={next_start}"

    print("  ✓ No overlapping regions detected")
    print()


def test_weight_distribution():
    """Test that weight distribution matches expectations."""
    text = """---
system: You are a helpful assistant.
---
User: Hello
Assistant: Hi there!
User: How are you?
Assistant: I'm doing well!"""

    regions = create_masking_regions(text)

    print("Test 4: Weight Distribution")

    # Count regions by role and mask status
    system_regions = [r for r in regions if r['role'] == 'system']
    user_regions = [r for r in regions if r['role'] == 'user']
    assistant_masked = [r for r in regions if r['role'] == 'assistant' and r['mask'] == True]
    assistant_unmasked = [r for r in regions if r['role'] == 'assistant' and r['mask'] == False]

    print(f"  System regions (should get system_weight): {len(system_regions)}")
    print(f"  User regions (should get user_weight): {len(user_regions)}")
    print(f"  Masked assistant regions (should get user_weight): {len(assistant_masked)}")
    print(f"  Unmasked assistant regions (should get assistant_weight): {len(assistant_unmasked)}")

    assert len(assistant_unmasked) == 1, \
        f"Should have exactly 1 unmasked assistant region, got {len(assistant_unmasked)}"

    assert len(assistant_masked) == 1, \
        f"Should have exactly 1 masked assistant region (context), got {len(assistant_masked)}"

    print("  ✓ Weight distribution correct")
    print()


def main():
    print("="*80)
    print("MASKING FIX VERIFICATION")
    print("="*80)
    print()

    try:
        test_single_assistant_turn()
        test_multiple_assistant_turns()
        test_no_overlapping_assignments()
        test_weight_distribution()

        print("="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print()
        print("The fix correctly ensures that:")
        print("  1. Only the FINAL assistant turn receives full assistant_weight")
        print("  2. All context assistant turns receive user_weight (like user turns)")
        print("  3. No tokens receive overlapping weight assignments")
        print()

    except AssertionError as e:
        print()
        print("="*80)
        print("TEST FAILED ✗")
        print("="*80)
        print(f"Error: {e}")
        print()
        sys.exit(1)


if __name__ == '__main__':
    main()
