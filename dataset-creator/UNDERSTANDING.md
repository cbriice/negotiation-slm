# Understanding Dataset Creator Architecture

## 1. JSONL Files in data/runs/bulk_data/real_training_data/

### File Locations
- `/home/carson/code/Intentia/dataset-creator/data/runs/bulk_data/real_training_data/150-same.jsonl`
- `/home/carson/code/Intentia/dataset-creator/data/runs/bulk_data/real_training_data/1906-diff.jsonl`
- `/home/carson/code/Intentia/dataset-creator/data/runs/bulk_data/real_training_data/1888-diff-2.jsonl`
- `/home/carson/code/Intentia/dataset-creator/data/runs/bulk_data/real_training_data/1880-diff-3.jsonl`

### JSONL Structure - SFT Training Format

Each line is a single JSON object representing one SFT training packet:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert negotiator."},
    {"role": "assistant", "content": "...first message from assistant (buyer or seller)..."},
    {"role": "user", "content": "...response from other party..."},
    {"role": "assistant", "content": "...next message from assistant..."},
    ...
  ],
  "system": {
    "assistant_style": "bluffing",           // personality tag of who is acting as assistant
    "difficulty": 0.31,                      // negotiation difficulty score (0.0-1.0)
    "key_strategy": "connection_appeal"      // negotiation strategy (can be empty)
  },
  "meta": {
    "opponent_style": "straightforward",     // personality tag of opponent
    "favorable_outcome": "favorable",        // "favorable", "unfavorable", or "neutral" from assistant's perspective
    "deal_closed": "true",                   // whether a deal was reached
    "rounds": 10                             // number of dialog turns
  }
}
```

### Key Observations

1. **Two Packets Per Negotiation**: Each negotiation creates 2 SFT packets:
   - One with BUYER as assistant
   - One with SELLER as assistant
   - Same dialog, different role assignments

2. **Personality Tags in Messages**:
   - The `messages` array contains alternating "assistant" and "user" roles
   - Role assignment depends on which party is the "assistant" for this packet
   - All real negotiations have personality pairings (e.g., "bluffing" vs "straightforward")

3. **Metadata Fields**:
   - `assistant_style`: The personality of whoever is playing the assistant role
   - `opponent_style`: The personality of the other party
   - `favorable_outcome`: Evaluated from the assistant's perspective
   - `difficulty`: Calculated from price gap between buyer_max and seller_min

---

## 2. Personality Classes and .tag Attributes

### Location
`/home/carson/code/Intentia/dataset-creator/prompts/personality_deck.py`

### All 12 Personalities with Tags

| Class Name | .tag Attribute | Description |
|------------|---|---|
| StraightToPoint | "straightforward" | Direct, wants to close quickly and favorably |
| Desperate | "desperation" | Will do almost anything to close the deal |
| Skeptic | "skepticism" | Not easily convinced to change mind |
| Bluffer | "bluffing" | Pushes strongly for favorable deal even if unreasonable |
| Confrontational | "confrontation" | Aggressively pushes for good deal using any means |
| Haggler | "haggling" | Doesn't fold easily, wants to extract maximum value |
| WeakEmotional | "concession" | Weak, wants to reach deal, dislikes confrontation |
| Rationalizer | "rationalization" | Uses reason and logic to win negotiation |
| StoneWall | "non-cooperation" | Doesn't want to hear counterparty, offer doesn't change |
| Opportunist | "opportunism" | Jumps at favorable deal as soon as it appears |
| HighHorse | "haughtiness" | High-and-mighty, thinks counterparty is ridiculous |
| Stubborn | "stubbornness" | Hates to change terms, needs lots of convincing |

### PersonalityTraits Dataclass

```python
@dataclass
class PersonalityTraits:
    # 15 core traits (0.0-1.0 range)
    assertiveness: float
    dominance: float
    confidence_self: float
    confidence_deal: float
    cooperativeness: float
    flexibility: float
    patience: float
    empathy: float
    rationality: float
    deceptiveness: float
    risk_tol: float
    persistence: float
    opportunism: float
    strategic: float
    composure: float
    
    # Independent parameters
    beta: float     # price threshold adjustment weight
    alpha: float    # (buyer only) initial offer = seller_min * alpha
    temp: float     # temperature for LLM generation
    
    # Trait-dependent parameters (computed in __post_init__)
    zeta: float     # max turns multiplier = patience
    c: float        # convergence parameter = sqrt(cooperativeness * flexibility)
    eta: float      # ultimatum parameter = sqrt(assertiveness * dominance * risk_tol)
    p: float        # concession probability = mean([1-confidence_self, cooperativeness, flexibility, 1-persistence])
```

### Example: Bluffer Personality

```python
class Bluffer():
    def __init__(self):
        self.description = "A bluffing negotiator..."
        self.tag = "bluffing"
        
        self.traits = PersonalityTraits(
            assertiveness=1,          # very assertive
            dominance=0.8,
            confidence_self=0.9,
            confidence_deal=0.2,      # low confidence in deal
            cooperativeness=0.3,      # not very cooperative
            flexibility=0.3,
            patience=0.5,
            empathy=0.4,
            rationality=0.4,
            deceptiveness=0.9,        # very deceptive
            risk_tol=0.9,            # high risk tolerance
            persistence=0.8,          # persistent
            opportunism=1,           # highly opportunistic
            strategic=1,             # strategic
            composure=0.8,
            beta=0.1,
            alpha=0.5,               # buyer opens at 50% of seller_min
            temp=0.7
        )
```

### How Personalities Load in PersonalityDeck

```python
class PersonalityDeck():
    def __init__(self):
        self.personalities = [
            lambda: StraightToPoint(),
            lambda: Desperate(),
            lambda: Skeptic(),
            lambda: Bluffer(),
            lambda: Confrontational(),
            lambda: Haggler(),
            lambda: WeakEmotional(),
            lambda: Rationalizer(),
            lambda: StoneWall(),
            lambda: Opportunist(),
            lambda: HighHorse(),
            lambda: Stubborn()
        ]
    
    def select_two_personalities(self, avoid_same: bool = True):
        """Select buyer and seller personalities, optionally avoiding same type."""
        if avoid_same and len(self.personalities) > 1:
            indices = random.sample(range(len(self.personalities)), 2)
            buyer = self.personalities[indices[0]]()
            seller = self.personalities[indices[1]]()
        else:
            # Same personality for both when avoid_same=False
            selected_personality = self.select_personality()
            personality_class = type(selected_personality)
            buyer = selected_personality
            seller = personality_class()
        
        return buyer, seller
```

---

## 3. SimulationRunner Architecture

### Location
`/home/carson/code/Intentia/dataset-creator/scripts/simulation_runner.py`

### Key Components

#### Initialization
```python
runner = SimulationRunner(
    seed_directory: str = "seeds/",
    model_name: str = "Qwen/Qwen2.5-14B-Instruct",
    max_turns: int = 25,
    seed_category: Optional[str] = None,
    seed_selection_method: str = "random",  # "random", "sequential", "all"
    specific_seed_file: Optional[str] = None,
    same_personality_count: int = 0         # 0=never, -1=always, N=first N runs
)
```

#### Main Workflow

1. **Load Seeds** (lines 73-100)
   - Loads negotiation seed files (product/service data with price ranges)
   - Uses MultiFileSeedLoader for directory or SeedLoader for single file

2. **Initialize Components** (lines 109-126)
   - Control function: Determines next action in negotiation
   - LLM interface: Generates responses using Qwen2.5-14B-Instruct
   - Personality deck: Selects buyer/seller personalities
   - Prompt manager: Manages action-specific prompts
   - Price dynamics: Calculates price adjustments

3. **Run Simulation** (lines 234-409)
   - For each negotiation:
     - Select seed (based on selection method)
     - Select unique personality pair (avoids duplicate pairings per seed)
     - Create negotiation packet
     - Run turn-by-turn negotiation

4. **Save Results** (lines 860-1040)
   - Saves flattened JSONL for training
   - Converts to SFT format automatically
   - Creates structured JSON with full details
   - Generates summary statistics

### Personality Pair Tracking

```python
# Track personality pairs used per seed to avoid duplicates
self.seed_personality_pairs = {}  # seed_id -> set of (buyer_pers, seller_pers) tuples

# In run_simulation():
used_pairs = self.seed_personality_pairs.get(seed.seed_id, set())

# Try up to 50 times to find a unique pair
while attempts < max_attempts:
    buyer_personality, seller_personality = self.personality_deck.select_two_personalities(
        avoid_same=(not use_same_personality)
    )
    pair_id = (buyer_personality.__class__.__name__, seller_personality.__class__.__name__)
    
    if pair_id not in used_pairs:
        valid_pair_found = True
        self.seed_personality_pairs[seed.seed_id].add(pair_id)
        break
```

### Turn-by-Turn Negotiation

**Forced Turns** (lines 448-482):
- Turn 0: Buyer opener (forced)
- Turn 1: Seller response (forced)
- Turn 2: Buyer initial price offer (forced)

**Control Function Turns** (Turn 3+, lines 484-831):
- Uses control function to determine next action
- Actions: "price_adjust", "accept", "reject", "clarify", "close", "extras", "ultimatum"
- Each action has personality-specific prompts

**Key Logic**:
- Prices converge automatically when gap < 1% of list price
- Negotiation ends on rejection unless other party persists
- Max turns = 20-25 by default

### Response Generation

```python
def _generate_response(self, packet: Dict, state: NegotiationState, action_prompt: str) -> str:
    # Build unified prompt from:
    # 1. System prompt (get_unified_prompt)
    # 2. Conversation history
    # 3. Action-specific prompt
    
    unified_prompt = get_unified_prompt(packet, state, action_prompt)
    
    # Generate using personality's temperature
    personality = packet["buyer_personality"] if is_buyer_turn else packet["seller_personality"]
    response = self.llm_interface.generate(
        unified_prompt,
        temperature=personality.traits.temp,
        sim_id=f"sim_{len(self.completed_negotiations)}"
    )
    
    return response.strip()
```

---

## 4. JSON Structure with assistant_style and opponent_style

### Source
`/home/carson/code/Intentia/dataset-creator/utils/sft_converter.py` (lines 209-282)

### Complete SFT Packet Structure

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert negotiator."
    },
    {
      "role": "assistant",
      "content": "Hi there, I'm looking to get a landing page copy written..."
    },
    {
      "role": "user",
      "content": "I appreciate the urgency, but my rate is $400..."
    },
    {
      "role": "assistant",
      "content": "Sure, I'm willing to recommend your services..."
    }
  ],
  "system": {
    "assistant_style": "bluffing",
    "difficulty": 0.31,
    "key_strategy": "connection_appeal"
  },
  "meta": {
    "seed_id": 3001,
    "opponent_style": "straightforward",
    "favorable_outcome": "favorable",
    "deal_closed": "true",
    "rounds": 10
  }
}
```

### Field Definitions

**messages** (List):
- Standard OpenAI chat format
- System role: Always "You are an expert negotiator."
- Alternating assistant/user roles based on which side is the "assistant"
- All dialog turns from negotiation

**system** (Dict):
- `assistant_style`: The .tag of the personality acting as assistant
  - Values: "straightforward", "desperation", "skepticism", "bluffing", "confrontation", "haggling", "concession", "rationalization", "non-cooperation", "opportunism", "haughtiness", "stubbornness"
- `difficulty`: Float 0.0-1.0 calculated from: `(seller_min - buyer_max) / (seller_list - seller_min)`
  - 0.0 = very easy (high overlap)
  - 1.0 = very hard (large gap)
- `key_strategy`: Negotiation strategy (often empty, can be filled by strategy labeler)
  - Examples: "anchor", "walk_away", "connection_appeal", "compromise", "market_logic"

**meta** (Dict):
- `seed_id`: Integer ID of the seed used
- `opponent_style`: The .tag of the opponent personality
- `favorable_outcome`: From assistant's perspective
  - "favorable": Assistant got a good deal
  - "unfavorable": Opponent got better deal
  - "neutral": No clear winner or no deal
- `deal_closed`: String "true" or "false"
- `rounds`: Integer number of dialog turns

### Two Packets Per Negotiation

The converter creates both perspectives:

**Packet 1 (Buyer as Assistant)**:
```json
{
  "system": {"assistant_style": "bluffing"},
  "meta": {"opponent_style": "straightforward", "favorable_outcome": "favorable"}
}
```

**Packet 2 (Seller as Assistant)**:
```json
{
  "system": {"assistant_style": "straightforward"},
  "meta": {"opponent_style": "bluffing", "favorable_outcome": "unfavorable"}
}
```

Note: Same messages, different role assignments!

### Conversion Process

```python
def create_sft_packets(self, negotiation: Dict) -> Tuple[Dict, Dict]:
    dialog = negotiation.get("dialog", [])
    buyer_personality = negotiation.get("buyer_personality")
    seller_personality = negotiation.get("seller_personality")
    
    buyer_tag = self.personality_tags.get(buyer_personality, "balanced")
    seller_tag = self.personality_tags.get(seller_personality, "balanced")
    
    difficulty = self.calculate_difficulty(negotiation)
    
    buyer_favorable = negotiation.get("buyer_favorable", False)
    seller_favorable = negotiation.get("seller_favorable", False)
    
    # Determine favorable outcome from each perspective
    favorable_outcome_buyer = "favorable" if buyer_favorable else ("unfavorable" if seller_favorable else "neutral")
    favorable_outcome_seller = "favorable" if seller_favorable else ("unfavorable" if buyer_favorable else "neutral")
    
    # Create buyer-as-assistant packet
    buyer_packet = {
        "messages": self.convert_dialog_to_messages(dialog, "BUYER"),
        "system": {
            "assistant_style": buyer_tag,
            "difficulty": difficulty,
            "key_strategy": ""
        },
        "meta": {
            "seed_id": negotiation.get("seed_id"),
            "opponent_style": seller_tag,
            "favorable_outcome": favorable_outcome_buyer,
            "deal_closed": str(negotiation.get("deal_closed", False)).lower(),
            "rounds": len(dialog)
        }
    }
    
    # Create seller-as-assistant packet
    seller_packet = {
        "messages": self.convert_dialog_to_messages(dialog, "SELLER"),
        "system": {
            "assistant_style": seller_tag,
            "difficulty": difficulty,
            "key_strategy": ""
        },
        "meta": {
            "seed_id": negotiation.get("seed_id"),
            "opponent_style": buyer_tag,
            "favorable_outcome": favorable_outcome_seller,
            "deal_closed": str(negotiation.get("deal_closed", False)).lower(),
            "rounds": len(dialog)
        }
    }
    
    return buyer_packet, seller_packet
```

---

## Summary Diagram

```
SimulationRunner
├── Load Seeds
│   └── seed.buyer_max, seed.seller_min, seed.seller_list, seed.item
│
├── Initialize Components
│   ├── PersonalityDeck (loads 12 personalities)
│   ├── LLMInterface (Qwen2.5-14B-Instruct)
│   ├── ControlFunction (determines next action)
│   └── PromptManager (action-specific prompts)
│
├── For Each Negotiation
│   ├── Select Seed
│   ├── Select Unique Personality Pair (avoids duplicates per seed)
│   │   ├── buyer_personality.__class__.__name__ (e.g., "Bluffer")
│   │   ├── seller_personality.__class__.__name__ (e.g., "StraightToPoint")
│   │   └── buyer_personality.tag (e.g., "bluffing")
│   │
│   ├── Run Turns (0-max_turns)
│   │   ├── Turn 0-2: Forced responses
│   │   ├── Turn 3+: Control function driven
│   │   └── Each turn: Generate response via LLM
│   │
│   └── Save Dialog
│       ├── negotiation.dialog = [(speaker, message), ...]
│       ├── negotiation.outcome = "deal" or "no_deal"
│       └── negotiation.final_price = X
│
├── Convert to SFT Format (SFTConverter)
│   ├── For Each Negotiation: Create 2 SFT Packets
│   │   ├── Packet 1: BUYER as assistant
│   │   │   └── system.assistant_style = "bluffing"
│   │   │   └── meta.opponent_style = "straightforward"
│   │   │
│   │   └── Packet 2: SELLER as assistant
│   │       └── system.assistant_style = "straightforward"
│   │       └── meta.opponent_style = "bluffing"
│   │
│   └── Write JSONL
│       └── Each line: 1 SFT packet JSON
│
└── Save Outputs
    ├── dataset_TIMESTAMP.jsonl (flattened)
    ├── dataset_TIMESTAMP_sft.jsonl (SFT format)
    ├── negotiations_structured_TIMESTAMP.json (full details)
    └── metadata.json + summary.txt
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `/prompts/personality_deck.py` | All 12 personalities with traits and tags |
| `/scripts/simulation_runner.py` | Main orchestrator (1234 lines) |
| `/utils/sft_converter.py` | Converts to SFT training format |
| `/seeds/` | Negotiation seed data (product/price info) |
| `/data/runs/bulk_data/real_training_data/` | Finished SFT training datasets |

