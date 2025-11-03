# Architecture Diagram and Data Flow

## System Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │   SIMULATION RUNNER                      │
                    │   (scripts/simulation_runner.py)         │
                    └─────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    v               v               v
            ┌──────────────┐ ┌────────────┐ ┌──────────────┐
            │ Seed Loader  │ │Personality │ │  LLM         │
            │ (seeds/)     │ │ Deck       │ │ Interface    │
            │              │ │ (12 types) │ │(Qwen2.5)     │
            └──────────────┘ └────────────┘ └──────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                    ┌───────────────v───────────────┐
                    │  FOR EACH NEGOTIATION         │
                    │  1. Select Seed               │
                    │  2. Select Personality Pair   │
                    │  3. Run Turns (0-max_turns)   │
                    │  4. Generate Responses        │
                    │  5. Save Dialog               │
                    └───────────────┬───────────────┘
                                    │
                                    v
                    ┌──────────────────────────────┐
                    │ Raw Dataset (flattened JSONL)│
                    │ - dialog (tuples)            │
                    │ - outcome                    │
                    │ - final_price                │
                    │ - personalities (names)      │
                    └──────────────┬───────────────┘
                                   │
                                   v
                    ┌──────────────────────────────┐
                    │ SFT Converter                │
                    │ (utils/sft_converter.py)     │
                    │                              │
                    │ Creates 2 packets per        │
                    │ negotiation:                 │
                    │ - Buyer as assistant        │
                    │ - Seller as assistant       │
                    └──────────────┬───────────────┘
                                   │
                                   v
                    ┌──────────────────────────────┐
                    │ SFT Training Dataset         │
                    │ (dataset_*_sft.jsonl)        │
                    │                              │
                    │ Each line:                   │
                    │ {                            │
                    │   "messages": [...],         │
                    │   "system": {...},           │
                    │   "meta": {...}              │
                    │ }                            │
                    └──────────────┬───────────────┘
                                   │
                                   v
                        Ready for SFT Training
                        (training-lab module)
```

## Negotiation Turn Flow

```
START
  │
  ├─ Turn 0 (FORCED): Buyer Opener
  │  └─ Control: Forced "opener"
  │     LLM: Generate opening pitch
  │
  ├─ Turn 1 (FORCED): Seller Response
  │  └─ Control: Forced "opener_response"
  │     LLM: Generate response to opening
  │
  ├─ Turn 2 (FORCED): Buyer Initial Price
  │  └─ Control: Forced "price_adjust"
  │     Action: Calculate initial offer using alpha
  │     LLM: Explain price offer
  │
  ├─ Turn 3+ (CONTROL FUNCTION DRIVEN)
  │  ├─ Evaluate Control Function
  │  │  ├─ Inputs: state, is_buyer
  │  │  └─ Outputs: action_probs, recommended_action
  │  │
  │  ├─ Sample Action from Distribution
  │  │  └─ Possible actions:
  │  │     ├─ price_adjust  (move price)
  │  │     ├─ accept        (take deal)
  │  │     ├─ reject        (refuse deal)
  │  │     ├─ clarify       (ask questions)
  │  │     ├─ close         (present final offer)
  │  │     ├─ extras        (add extras)
  │  │     └─ ultimatum     (final demand)
  │  │
  │  ├─ Handle Forced Responses
  │  │  ├─ If close → next turn must be accept/reject
  │  │  │  (uses personality.p to decide)
  │  │  │
  │  │  └─ If clarify → next turn must be clarify_respond
  │  │
  │  ├─ Get Action-Specific Prompt
  │  │  └─ Prompt Manager selects prompt for action
  │  │     (varies by personality)
  │  │
  │  └─ Generate Response
  │     └─ LLM with personality.temp
  │
  ├─ Update State
  │  ├─ Update prices if price_adjust
  │  ├─ Check for price overlap (deal convergence)
  │  ├─ Check for rejection (end negotiation?)
  │  └─ Continue to next turn
  │
  └─ End Conditions
     ├─ Deal accepted
     ├─ Both parties rejected
     ├─ Price overlap detected
     ├─ Max turns reached
     └─ RETURN negotiation packet

```

## Personality Selection and Tracking

```
PERSONALITY DECK (12 total)
├─ StraightToPoint (straightforward)
├─ Desperate (desperation)
├─ Skeptic (skepticism)
├─ Bluffer (bluffing)
├─ Confrontational (confrontation)
├─ Haggler (haggling)
├─ WeakEmotional (concession)
├─ Rationalizer (rationalization)
├─ StoneWall (non-cooperation)
├─ Opportunist (opportunism)
├─ HighHorse (haughtiness)
└─ Stubborn (stubbornness)

FOR EACH SEED:
┌────────────────────────────┐
│ seed_personality_pairs     │
│ {seed_id: set of pairs}    │
├────────────────────────────┤
│ seed_1: {                  │
│   (Bluffer, Skeptic),      │ ← Used in negotiation 1
│   (Desperate, Haggler),    │ ← Used in negotiation 2
│   (StraightToPoint, ...)   │ ← Used in negotiation 3
│ }                          │
└────────────────────────────┘

SELECTION LOGIC:
  while attempts < 50:
    candidate_pair = random_select_2(personalities)
    if candidate_pair NOT in used_pairs[seed_id]:
      ✓ VALID - use this pair
      break
    attempts += 1

This ensures NO duplicate personality pairings per seed!
```

## Data Transformation Pipeline

```
RAW NEGOTIATION OUTPUT
├─ dialog: [(BUYER, msg), (SELLER, msg), ...]
├─ buyer_personality: "Bluffer" (class name)
├─ seller_personality: "StraightToPoint" (class name)
├─ seed_id: 3001
├─ outcome: "deal" / "no_deal"
├─ final_price: 189.50
├─ buyer_favorable: true/false
└─ seller_favorable: true/false

                    │
                    │ SFTConverter.create_sft_packets()
                    │
                    v

PACKET 1: Buyer as Assistant
├─ messages:
│  ├─ {role: "system", content: "You are an expert negotiator."}
│  ├─ {role: "assistant", content: "Bluffer's opening..."}
│  ├─ {role: "user", content: "StraightToPoint's response..."}
│  ├─ {role: "assistant", content: "Bluffer's counter..."}
│  └─ ... (alternating assistant/user)
├─ system:
│  ├─ assistant_style: "bluffing"
│  ├─ difficulty: 0.31
│  └─ key_strategy: "" (can be filled by labeler)
└─ meta:
   ├─ seed_id: 3001
   ├─ opponent_style: "straightforward"
   ├─ favorable_outcome: "favorable"
   ├─ deal_closed: "true"
   └─ rounds: 10

PACKET 2: Seller as Assistant
├─ messages:
│  ├─ {role: "system", content: "You are an expert negotiator."}
│  ├─ {role: "assistant", content: "StraightToPoint's response..."}
│  ├─ {role: "user", content: "Bluffer's opening..."}
│  ├─ {role: "assistant", content: "StraightToPoint's counter..."}
│  └─ ... (SAME dialog, DIFFERENT roles)
├─ system:
│  ├─ assistant_style: "straightforward"
│  ├─ difficulty: 0.31
│  └─ key_strategy: ""
└─ meta:
   ├─ seed_id: 3001
   ├─ opponent_style: "bluffing"
   ├─ favorable_outcome: "unfavorable"
   ├─ deal_closed: "true"
   └─ rounds: 10
```

## Control Function State Flow

```
NEGOTIATION STATE
├─ r: turn number (0-25)
├─ Ps: seller's reservation price (min acceptable)
├─ Pb: buyer's reservation price (max acceptable)
├─ buyer_offer: current buyer's price
├─ seller_offer: current seller's price
├─ P_k: midpoint price
├─ n_b_k: buyer extras count
├─ n_s_k: seller extras count
└─ l_k: extras included count

PRICE STATE
├─ buyer_price: current buyer price
├─ seller_price: current seller price
├─ initial_buyer_price: first buyer offer
├─ initial_seller_price: always seller_list
├─ current_price: most recent offer
└─ turn_number: current turn

                    │
                    v
                
        CONTROL FUNCTION EVALUATION
        ├─ Calculate features:
        │  ├─ cg: buyer goodwill
        │  ├─ cz: buyer pressure
        │  ├─ s: buyer satisfaction
        │  └─ A_k: extra items available
        │
        ├─ Get action probabilities:
        │  ├─ price_adjust: 0.4
        │  ├─ accept: 0.2
        │  ├─ reject: 0.15
        │  ├─ clarify: 0.1
        │  ├─ close: 0.1
        │  ├─ extras: 0.05
        │  └─ ultimatum: 0.0 (contextual)
        │
        └─ Sample action using probabilities
           └─ ACTION SELECTED
```

## File Organization

```
dataset-creator/
├─ prompts/
│  └─ personality_deck.py
│     ├─ PersonalityTraits (dataclass)
│     ├─ 12 Personality Classes
│     └─ PersonalityDeck (selector)
│
├─ scripts/
│  ├─ simulation_runner.py (1234 lines)
│  │  ├─ SimulationRunner (main orchestrator)
│  │  ├─ run_simulation()
│  │  ├─ _run_single_negotiation()
│  │  ├─ _select_action_and_prompt()
│  │  ├─ _generate_response()
│  │  └─ save_results()
│  │
│  └─ utils/
│     ├─ seed_loader.py
│     ├─ control_function.py
│     ├─ prompt_manager.py
│     ├─ llm_interface.py
│     ├─ price_dynamics.py
│     └─ sft_converter.py
│
├─ seeds/
│  ├─ tech_seeds.json
│  ├─ vehicle_seeds.json
│  ├─ service_seeds.json
│  └─ travel_seeds.json
│
└─ data/runs/
   ├─ run_TIMESTAMP/
   │  ├─ dataset_TIMESTAMP.jsonl (raw)
   │  ├─ dataset_TIMESTAMP_sft.jsonl (SFT)
   │  ├─ negotiations_structured_TIMESTAMP.json
   │  ├─ metadata.json
   │  └─ summary_TIMESTAMP.txt
   │
   └─ bulk_data/real_training_data/
      ├─ 150-same.jsonl
      ├─ 1906-diff.jsonl
      ├─ 1888-diff-2.jsonl
      └─ 1880-diff-3.jsonl
```

## Message Role Assignment Logic

```
SAME DIALOG - DIFFERENT ROLE ASSIGNMENTS

Original negotiation:
turn 0: BUYER says "Opening pitch..."
turn 1: SELLER says "Response..."
turn 2: BUYER says "Price: $X"
turn 3: SELLER says "Counter: $Y"

                    │
        ┌───────────┴───────────┐
        │                       │
        v                       v

PACKET (Buyer as assistant):    PACKET (Seller as assistant):

{role: "system", ...}           {role: "system", ...}
{role: "assistant", ...}        {role: "user", ...}      ← Same content,
{role: "user", ...}    ←─────→  {role: "assistant", ...} ← Different roles
{role: "assistant", ...         {role: "user", ...}
{role: "user", ...}             {role: "assistant", ...}
```

## Personality Traits Calculation Example (Bluffer)

```
DEFINED TRAITS (from code)
├─ assertiveness: 1.0      ─┐
├─ dominance: 0.8           │
├─ confidence_self: 0.9     ├─ Direct from class definition
├─ confidence_deal: 0.2     │
├─ ... (12 more traits)     │
└─ risk_tol: 0.9          ─┘

POST_INIT CALCULATIONS
├─ zeta = patience
│         = 0.5
│
├─ c = sqrt(cooperativeness * flexibility)
│    = sqrt(0.3 * 0.3)
│    = sqrt(0.09)
│    = 0.3
│
├─ eta = sqrt(assertiveness * dominance * risk_tol)
│      = sqrt(1.0 * 0.8 * 0.9)
│      = sqrt(0.72)
│      = 0.849 ✓ (> 0.5, so can use ultimatums)
│
└─ p = mean([1-confidence_self, cooperativeness, flexibility, 1-persistence])
     = mean([0.1, 0.3, 0.3, 0.2])
     = 0.225 (22.5% chance of conceding on price_adjust)
```

