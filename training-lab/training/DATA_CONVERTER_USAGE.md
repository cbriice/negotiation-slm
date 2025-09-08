# Data Converter CLI Usage

The `data_converter.py` module provides a command-line interface for converting ParlAI negotiation episodes into training formats (SFT and DPO).

## Installation

Ensure ParlAI is installed:
```bash
pip install parlai
```

## Commands

### 1. Convert Episodes to SFT Format
```bash
python -m training.data_converter sft --input episodes.jsonl --output sft_data.jsonl
```
Options:
- `--history-window`: Number of recent turns for context (default: 5)

### 2. Create DPO Pairs
```bash
python -m training.data_converter dpo --good good_episodes.jsonl --bad bad_episodes.jsonl
```
Options:
- `--output`: Output file path
- `--max-pairs`: Maximum number of pairs to generate

### 3. Auto-Generate DPO Pairs
Automatically pairs episodes based on quality scores:
```bash
python -m training.data_converter auto-dpo --input episodes.jsonl --threshold 5.0
```

### 4. Export Built-in Tasks
Export Deal-or-No-Deal and Casino tasks:
```bash
python -m training.data_converter builtin --tasks dealnodeal casino --max-episodes 1000
```
Options:
- `--no-sft`: Skip SFT conversion

### 5. Score Episodes
```bash
python -m training.data_converter score --input episodes.jsonl --output scored.jsonl
```

### 6. Full Pipeline
Prepare complete training dataset:
```bash
python -m training.data_converter prepare --input raw_episodes.jsonl --output-dir data/prepared
```

This will:
1. Score all episodes
2. Convert to SFT format
3. Filter high-quality examples
4. Generate DPO pairs

## Output Structure

```
data/training/
├── sft/                 # SFT format data
├── dpo/                 # DPO training pairs
├── builtin/             # Exported built-in tasks
├── scored/              # Episodes with quality scores
└── prepared/            # Full pipeline output
```

## Examples

### Export and prepare built-in tasks for training:
```bash
# Export tasks
python -m training.data_converter builtin

# Score and prepare exported episodes
python -m training.data_converter prepare --input data/training/builtin/dealnodeal_episodes.jsonl
```

### Process custom negotiation episodes:
```bash
# Score episodes
python -m training.data_converter score --input negotiations.jsonl

# Generate training data
python -m training.data_converter prepare --input scored_episodes.jsonl
```