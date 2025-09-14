#!/bin/bash

# Script to test SFT trainer functionality without running actual training

echo "========================================"
echo "SFT Trainer Test Suite"
echo "========================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Note: The test script will auto-clear its log file (logs/test_sft_trainer.log)
# This tee output is separate for console review

# Run unit tests
echo "Running unit tests..."
python tests/test_sft_trainer.py 2>&1 | tee logs/sft_test_console_output.log

echo ""
echo "========================================"
echo "Test results saved to:"
echo "  - logs/test_sft_trainer.log (detailed)"
echo "  - logs/sft_test_console_output.log (console mirror)"
echo ""

# Run dry-run test only
echo "Running dry-run test only..."
python tests/test_sft_trainer.py --dry-run

echo ""
echo "========================================"
echo "All tests completed!"
echo "Note: Log files auto-clear on each run to prevent bloat"
echo "========================================"