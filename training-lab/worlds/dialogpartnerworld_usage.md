# DialogPartnerWorld Usage Guide

## Overview
DialogPartnerWorld manages negotiation episodes between an OllamaNegotiator agent and either a RuleAgent or human partner. It supports two-part response generation (strategy → response) and comprehensive metrics tracking.

## Key Components

### Initialization
```python
from worlds.dialog_world import DialogPartnerWorld

world = DialogPartnerWorld(
    opt={
        'partner_mode': 'rule',  # 'rule' for RuleAgent, 'human' for console
        'max_turns': 20,
        'batch_size': 10,
        'use_two_part_response': True  # Enable strategy generation
    },
    agents=[negotiator_agent, partner_agent]  # or [negotiator_agent] for human mode
)
```

### Running Episodes

#### Single Episode
```python
episode = world.run_episode()
# Returns dict with dialogue, strategies, metrics, and outcome
```

#### Batch Execution
```python
results = world.run_batch(num_episodes=100)
# Automatically saves episodes and metrics to data/dialogues/
```

## Partner Modes

### Rule Mode
- Uses RuleAgent with randomized personas
- Enforces negotiation constraints
- Can terminate on protocol violations
- Personas include: aggressive, sympathetic, analytical, opportunistic, collaborative

### Human Mode  
- Interactive console negotiation
- Human plays as partner via text input
- Useful for testing edge cases

## Two-Part Response Flow

When `use_two_part_response=True`:
1. **Strategy Generation**: Agent creates JSON strategy with action_indicator
2. **Response Generation**: Action-specific prompt generates natural response

Supported actions: PROPOSE, COMMIT, ACK, NACK, NO_DEAL

## Episode Structure

Each episode contains:
- `dialogue_id`: Unique identifier
- `state`: Item details, price constraints
- `dialogue`: Turn-by-turn messages with actions
- `strategies`: Agent's internal strategies (if two-part enabled)
- `outcome`: success/failure/timeout
- `metrics`: Tokens, turns, timing, action distribution

## Data Storage

Episodes saved to:
- `data/dialogues/batch_{timestamp}/episodes_{batch_num}.jsonl`
- `data/dialogues/batch_{timestamp}/metrics.csv`
- `data/dialogues/batch_{timestamp}/session_summary.json`

## Metrics Tracked

Per-episode:
- Token counts (agent/partner)
- Turn counts
- Deal success/failure
- Terms agreement percentage
- Action distribution

Session-level:
- Success rate
- Average tokens/turns
- Strategy effectiveness

## Quick Start

```python
# Basic test run
from worlds.dialog_world import DialogPartnerWorld
from agents.ollama_wrapper import OllamaNegotiator
from agents.rule_agent import RuleAgent

# Create agents
negotiator = OllamaNegotiator(opt={'model_name': 'mistral:7b'})
rule_partner = RuleAgent(opt={})

# Create world
world = DialogPartnerWorld(
    opt={'partner_mode': 'rule', 'use_two_part_response': True},
    agents=[negotiator, rule_partner]
)

# Run episodes
results = world.run_batch(num_episodes=10)
print(f"Success rate: {results['session_metrics']['success_rate']:.2%}")
```

## Testing Commands

```bash
# Run with rule partner
python -m worlds.dialog_world --partner-mode rule --num-episodes 10

# Interactive human mode
python -m worlds.dialog_world --partner-mode human --num-episodes 1

# With two-part response
python -m worlds.dialog_world --use-two-part-response --num-episodes 5
```