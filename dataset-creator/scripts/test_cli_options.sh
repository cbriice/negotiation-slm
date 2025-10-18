#!/bin/bash

echo "=== Testing Simulation Runner CLI Options ==="
echo ""

# Show help
echo "1. Showing help information:"
echo "----------------------------"
python simulation_runner.py --help
echo ""

# Example commands (commented out to avoid actual execution)
echo "2. Example commands to run simulations with different configurations:"
echo "--------------------------------------------------------------------"

echo "# Run with random seed selection (default), data saved to timestamped directory:"
echo "python simulation_runner.py --num-negotiations 5"
echo ""

echo "# Run with sequential seed selection:"
echo "python simulation_runner.py --seed-selection sequential --num-negotiations 5"
echo ""

echo "# Run with all seeds in a directory:"
echo "python simulation_runner.py --seed-selection all"
echo ""

echo "# Filter by category (e.g., tech):"
echo "python simulation_runner.py --seed-category tech --num-negotiations 3"
echo ""

echo "# Use a specific seed file:"
echo "python simulation_runner.py --seed-file seeds/tech_seeds.jsonl --num-negotiations 2"
echo ""

echo "# Use a specific seed by ID from a file:"
echo "python simulation_runner.py --seed-file seeds/tech_seeds.jsonl --seed-id 1001"
echo ""

echo "# Configure model and custom output directory:"
echo "python simulation_runner.py --model Qwen/Qwen2.5-14B-Instruct --output-dir data/custom_runs/"
echo ""

echo "# Run without saving structured data (only JSONL for training):"
echo "python simulation_runner.py --no-structured --num-negotiations 10"
echo ""

echo "# Full configuration example:"
echo "python simulation_runner.py \\"
echo "  --seed-dir seeds/ \\"
echo "  --seed-category tech \\"
echo "  --seed-selection sequential \\"
echo "  --num-negotiations 10 \\"
echo "  --max-turns 30 \\"
echo "  --model Qwen/Qwen2.5-14B-Instruct \\"
echo "  --output-dir data/tech_runs/"
echo ""

echo "=== Configuration Summary ==="
echo ""
echo "Seed Configuration Options:"
echo "  --seed-dir          : Directory containing seed files (default: seeds/)"
echo "  --seed-file         : Use a specific seed file instead of directory"
echo "  --seed-category     : Filter seeds by category (e.g., tech, furniture)"
echo "  --seed-id           : Use specific seed by ID (requires --seed-file)"
echo "  --seed-selection    : Selection method: random, sequential, or all"
echo ""
echo "Simulation Options:"
echo "  --num-negotiations  : Number of negotiations to run (default: 2)"
echo "  --max-turns         : Maximum turns per negotiation (default: 20)"
echo "  --model             : LLM model to use (default: llama3.3:70b)"
echo ""
echo "Output Options:"
echo "  --output-dir        : Base output directory for runs (default: data/runs/)"
echo "  --no-structured     : Skip saving structured version of data"
echo ""
echo "Output Structure:"
echo "  Each run creates a timestamped directory (e.g., run_20240118_142305/) containing:"
echo "  - metadata.json                     : Run configuration and summary"
echo "  - dataset_TIMESTAMP.jsonl          : Flattened dataset for training"
echo "  - negotiations_structured_TIMESTAMP.json : Structured data with full details"
echo "  - summary_TIMESTAMP.txt            : Human-readable summary statistics"