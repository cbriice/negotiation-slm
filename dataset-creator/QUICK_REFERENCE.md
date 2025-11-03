# Quick Reference Guide

## File Locations

| What | Where |
|------|-------|
| Personalities | `/prompts/personality_deck.py` (12 classes) |
| Simulation Runner | `/scripts/simulation_runner.py` (1234 lines) |
| SFT Converter | `/utils/sft_converter.py` |
| Training Data | `/data/runs/bulk_data/real_training_data/*.jsonl` |
| Seeds | `/seeds/*.json` |

## The 12 Personalities (with .tag values)

```
1. StraightToPoint     -> "straightforward"
2. Desperate           -> "desperation"
3. Skeptic             -> "skepticism"
4. Bluffer             -> "bluffing"
5. Confrontational     -> "confrontation"
6. Haggler             -> "haggling"
7. WeakEmotional       -> "concession"
8. Rationalizer        -> "rationalization"
9. StoneWall           -> "non-cooperation"
10. Opportunist        -> "opportunism"
11. HighHorse          -> "haughtiness"
12. Stubborn           -> "stubbornness"
```

## PersonalityTraits Structure

**15 Core Traits** (0.0-1.0):
assertiveness, dominance, confidence_self, confidence_deal, cooperativeness, flexibility, patience, empathy, rationality, deceptiveness, risk_tol, persistence, opportunism, strategic, composure

**3 Direct Parameters**:
- `beta`: Price adjustment weight
- `alpha`: Buyer's initial offer multiplier (buyer only)
- `temp`: LLM temperature for generation

**4 Computed Parameters** (in __post_init__):
- `zeta` = patience
- `c` = sqrt(cooperativeness * flexibility)
- `eta` = sqrt(assertiveness * dominance * risk_tol)
- `p` = mean([1-confidence_self, cooperativeness, flexibility, 1-persistence])

## SFT Packet Structure

```json
{
  "messages": [ ... ],  // Chat format: alternating assistant/user
  "system": {
    "assistant_style": "bluffing",           // Which personality is assistant
    "difficulty": 0.31,                      // Negotiation difficulty (0.0-1.0)
    "key_strategy": "connection_appeal"      // Optional strategy label
  },
  "meta": {
    "seed_id": 3001,                         // Seed ID
    "opponent_style": "straightforward",     // Opponent personality
    "favorable_outcome": "favorable",        // "favorable"/"unfavorable"/"neutral"
    "deal_closed": "true",                   // String boolean
    "rounds": 10                             // Number of dialog turns
  }
}
```

## Personality Tags vs Class Names

```python
personality = Bluffer()
personality.__class__.__name__  # Returns "Bluffer"
personality.tag                 # Returns "bluffing"
```

## Two Packets Per Negotiation

Same dialog, different perspectives:

```
Negotiation: Bluffer (buyer) vs StraightToPoint (seller)

Packet 1 (Buyer = assistant):
  system.assistant_style = "bluffing"
  meta.opponent_style = "straightforward"
  meta.favorable_outcome = "favorable"

Packet 2 (Seller = assistant):
  system.assistant_style = "straightforward"
  meta.opponent_style = "bluffing"
  meta.favorable_outcome = "unfavorable"
```

## SimulationRunner Workflow

```
1. Load Seeds        <- Product/price data
2. Initialize        <- PersonalityDeck, LLMInterface, ControlFunction
3. For N negotiations:
   a. Select seed
   b. Select unique personality pair (avoids duplicates per seed)
   c. Run turns 0-max_turns:
      - Turns 0-2: Forced (opener, response, initial offer)
      - Turns 3+: Control function driven (action selection)
      - Each turn: Generate response via LLM
   d. Save dialog, outcome, price
4. Convert to SFT format (creates 2 packets per negotiation)
5. Save outputs:
   - dataset_TIMESTAMP.jsonl
   - dataset_TIMESTAMP_sft.jsonl
   - negotiations_structured_TIMESTAMP.json
   - metadata.json
   - summary_TIMESTAMP.txt
```

## Personality Selection

```python
# Select 2 different personalities
buyer, seller = personality_deck.select_two_personalities(avoid_same=True)

# Select 2 same personalities (can be forced with runner option)
buyer, seller = personality_deck.select_two_personalities(avoid_same=False)

# Track to avoid duplicates per seed
seed_personality_pairs[seed_id].add((buyer_name, seller_name))
```

## Key Formulas

**Difficulty Calculation**:
```
difficulty = (seller_min - buyer_max) / (seller_list - seller_min)
0.0 = easy (overlap exists)
1.0 = hard (large gap)
```

**Buyer's Initial Offer**:
```
initial_buyer_offer = seller_min * alpha
Example: Bluffer.alpha = 0.5, so opens at 50% of seller_min
```

**Concession Probability**:
```
p = mean([1-confidence_self, cooperativeness, flexibility, 1-persistence])
Used to decide if party will concede on price_adjust turns
```

**Ultimatum Parameter**:
```
eta = sqrt(assertiveness * dominance * risk_tol)
If eta > 0.5: personality can use ultimatums
```

## Important Constants

- **Max turns**: 20-25 (configurable)
- **Model**: Qwen/Qwen2.5-14B-Instruct
- **Convergence threshold**: 1% of list price
- **Max personality pair attempts**: 50
- **System prompt**: "You are an expert negotiator."

## Data Flow

```
Seeds + Personalities
        |
        v
SimulationRunner
   (turn-by-turn)
        |
        v
Raw Dataset (flattened JSONL)
   dialog + outcome + price
        |
        v
SFTConverter
(2 packets per negotiation)
        |
        v
SFT Training Dataset (JSONL)
   messages + system + meta
```

## Common Commands

```bash
# Run simulation with 10 negotiations
python scripts/simulation_runner.py --num-negotiations 10

# Run with specific seed file
python scripts/simulation_runner.py --seed-file seeds/tech_seeds.json

# Force same personality for all runs
python scripts/simulation_runner.py --same-personality

# Force same personality for first 5 runs
python scripts/simulation_runner.py --same-personality 5

# Use sequential seed selection instead of random
python scripts/simulation_runner.py --seed-selection sequential

# Run max 15 turns
python scripts/simulation_runner.py --max-turns 15
```

## File Size Reference

- Typical JSONL line: 1-3 KB
- 100 negotiations: ~200-300 KB (100 lines in flattened, 200 in SFT)
- Full dialog turn count: Usually 8-15 turns

## Strategy Labels (in key_strategy field)

Common values found in datasets:
- "anchor" - Setting initial price anchor
- "walk_away" - Threatening to leave
- "connection_appeal" - Appeal based on relationship
- "compromise" - Meeting in middle
- "market_logic" - Using market justification
- Empty string if not yet labeled

