"""Integration test for curriculum training."""

import subprocess
import sys

def test_curriculum_training():
    """Test that curriculum training runs without errors."""
    cmd = [
        sys.executable, "train_curriculum.py",
        "--test-mode",
        "--max-steps", "20",
        "--batch-size", "2"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, f"Training failed: {result.stderr}"
    assert "Phase 1" in result.stdout
    assert "Curriculum training complete" in result.stdout

    print("✓ Curriculum training integration test passed!")

if __name__ == "__main__":
    test_curriculum_integration()
