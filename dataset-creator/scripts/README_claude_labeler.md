# Claude Key Strategy Labeler

Automated script for labeling negotiation strategies in SFT packets using Claude API.

## Setup

1. Install required dependencies:
```bash
pip install anthropic python-dotenv tqdm
```

2. Set up your API key in `.env`:
```bash
ANTHROPIC_API_KEY=your_api_key_here
CLAUDE_MODEL=claude-sonnet-4-5-20250929  # Optional, can override with --model
```

## Usage

### Basic Usage
```bash
python scripts/claude_key_strategy_labeler.py data/runs/bulk_data/training_data/dataset_20251026_114958_sft.jsonl
```

This will:
- Process the input JSONL file
- Call Claude API to label each negotiation's key strategy
- Save output to `dataset_20251026_114958_sft_labeled.jsonl`

### Advanced Options
```bash
# Specify output file
python scripts/claude_key_strategy_labeler.py input.jsonl -o output_labeled.jsonl

# Use a specific model
python scripts/claude_key_strategy_labeler.py input.jsonl -m claude-sonnet-4-5-20250929

# Adjust batch size (saves every N packets)
python scripts/claude_key_strategy_labeler.py input.jsonl -b 20
```

### Testing
Run the test script to verify setup:
```bash
python scripts/test_claude_labeler.py
```

## How It Works

1. **Input Processing**: Reads SFT packets from JSONL file
2. **Strategy Detection**: For each packet:
   - Extracts negotiation messages
   - Formats them as "ASSISTANT: ..." and "USER: ..."
   - Sends to Claude with the labeling prompt
   - Claude returns one of 9 strategy tags
3. **Output**: Updates `system.key_strategy` field and saves to new file

## Strategy Tags

The script labels negotiations with these strategies:
- `anchor` - Stands firm at one price
- `compromise` - Works towards mutual agreement
- `market_logic` - Uses market context to persuade
- `connection_appeal` - Offers connections as incentive
- `empathy_appeal` - Appeals to counterparty's empathy
- `aggro` - Aggressive tactics for favorable outcome
- `concede` - Concedes to close the deal
- `walk_away` - Exits when favorable deal unlikely
- `flatter` - Uses flattery to get better terms

## Features

- **Batch Processing**: Saves progress periodically
- **Error Handling**: Retries failed API calls with exponential backoff
- **Skip Existing**: Won't re-label packets that already have key_strategy
- **Progress Tracking**: Shows progress bar and statistics
- **Rate Limiting**: Built-in delays to respect API limits

## Output Format

The script updates the `key_strategy` field in each packet:
```json
{
    "messages": [...],
    "system": {
        "assistant_style": "non-cooperation",
        "difficulty": 0.4,
        "key_strategy": "anchor"  // Added by this script
    },
    "meta": {...}
}
```

## Statistics

After processing, displays:
- Total packets processed
- Successfully labeled
- Failed to label
- Skipped (already labeled)
- Success rate percentage