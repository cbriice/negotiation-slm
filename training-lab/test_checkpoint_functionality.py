#!/usr/bin/env python3
"""
Quick test to verify checkpoint functionality works.
"""

import sys
from pathlib import Path

print("Testing checkpoint functionality...\n")

# Test 1: Import checkpoint_utils
print("1. Testing imports...")
try:
    from training.curriculum.checkpoint_utils import (
        adjust_phase_ranges,
        print_resume_banner,
        validate_phase_compatibility
    )
    print("   ✓ checkpoint_utils imports OK")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Test phase range adjustment
print("\n2. Testing phase range adjustment...")
try:
    test_phases = [
        {"name": "phase1", "step_start": 0, "step_end": 800},
        {"name": "phase2", "step_start": 800, "step_end": 2100},
        {"name": "phase3", "step_start": 2100, "step_end": 3000}
    ]

    adjusted = adjust_phase_ranges(test_phases, 2)

    # Verify adjustment
    assert len(adjusted) == 2, f"Expected 2 phases, got {len(adjusted)}"
    assert adjusted[0]["step_start"] == 0, f"Phase 2 should start at 0, got {adjusted[0]['step_start']}"
    assert adjusted[0]["step_end"] == 1300, f"Phase 2 should end at 1300, got {adjusted[0]['step_end']}"
    assert adjusted[1]["step_start"] == 1300, f"Phase 3 should start at 1300, got {adjusted[1]['step_start']}"
    assert adjusted[1]["step_end"] == 2200, f"Phase 3 should end at 2200, got {adjusted[1]['step_end']}"

    print("   ✓ Phase range adjustment works correctly")
    print(f"     Original: Phase 1 (0-800), Phase 2 (800-2100), Phase 3 (2100-3000)")
    print(f"     Adjusted (from phase 2): Phase 2 (0-1300), Phase 3 (1300-2200)")
except Exception as e:
    print(f"   ✗ Phase adjustment failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test validation (should just warn, not error)
print("\n3. Testing phase compatibility validation...")
try:
    metadata = {
        "phase_number": 1,
        "phase_name": "phase1",
        "step": 800,
        "total_phases": 3
    }
    validate_phase_compatibility(metadata, test_phases, 2)
    print("   ✓ Validation runs without errors")
except Exception as e:
    print(f"   ✗ Validation failed: {e}")
    sys.exit(1)

# Test 4: Test banner printing
print("\n4. Testing resume banner...")
try:
    mock_path = Path("models/curriculum/phase_1_step_800")
    mock_metadata = {
        "phase_number": 1,
        "step": 800,
        "total_phases": 3
    }
    print_resume_banner(mock_path, mock_metadata, adjusted)
    print("   ✓ Banner prints successfully")
except Exception as e:
    print(f"   ✗ Banner printing failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("All checkpoint functionality tests passed!")
print("="*60)
