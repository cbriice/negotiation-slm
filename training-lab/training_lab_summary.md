# Training Lab - ParlAI Worlds & SFT Pipeline Reference

## SelfChatWorld Implementation

### Core File: `worlds/selfchat_world.py`

**Purpose**: Generates training data through agent self-play negotiations

**Initialization**:
```python
world = SelfChatWorld(
    opt={
        'num_episodes': 1500,
        'batch_size': 16,  # For parallel GPU inference
        'model_name': 'mistral-7b',
        'temperature': 0.5
    },
    agents=[buyer_agent, seller_agent],  # Both OllamaNegotiator instances
    teacher=SelfplayTeacher()  # Provides state seeds
)
```

**Key Methods**:

- **`run_episode(seed, episode_id)`**: Executes single negotiation
  - Initializes agents with buyer/seller states from seed
  - Manages turn-taking until COMMIT+ACK or failure
  - Tracks all metrics (tokens, turns, violations)
  - Returns complete episode as JSONL dict

- **`run_parallel_episodes(seeds, start_id)`**: Batch processing
  - Groups episodes by batch_size for GPU efficiency
  - Maintains conversation contexts across turns
  - Handles partial batches gracefully
  - Returns list of episode dicts

- **`parley()`**: ParlAI interface method
  - Called by ParlAI framework
  - Orchestrates episode execution
  - Manages session-level metrics aggregation

**Episode Output Structure**:
```json
{
  "episode_id": "ep_0001",
  "dialogue_id": "selfchat_session_001_ep_0001",
  "seed": {...},  // Original state seed
  "turns": [
    {
      "agent": "buyer",
      "strategy": {...},  // Internal strategy JSON
      "message": "...",   // Public message (no action tags)
      "action": "PROPOSE",  // Backend-only tag
      "tokens": {"strategy": 45, "response": 89}
    }
  ],
  "outcome": {
    "deal_closed": true,
    "final_price": 950,
    "terms": {...}  // Final TermsJSON if agreed
  },
  "metrics": {
    "total_turns": 8,
    "buyer_tokens": 412,
    "seller_tokens": 389,
    "duration_seconds": 23.5,
    "violations": []
  }
}
```

**State Seed Selection**:
- Teacher provides seeds from 5 categories
- 10% are "regression seeds" (maintained across runs)
- Mutations applied: price jitter (±20%), item defects, logistics changes

## DialogPartnerWorld Implementation

### Core File: `worlds/dialog_world.py`

**Purpose**: Training against RuleAgent with diverse personas or human evaluation

**Initialization**:
```python
world = DialogPartnerWorld(
    opt={'mode': 'rule'},  # or 'human' for console
    agents=[negotiator_agent, rule_agent],
    teacher=RuleOpponentTeacher()
)
```

**RuleAgent Personas** (7 types with distinct behaviors):
- **Aggressive**: Low initial offers, quick escalation
- **Cooperative**: Fair pricing, flexible terms
- **Deceptive**: Misleading claims, bundle manipulation
- **Sympathetic**: Emotional appeals, sob stories
- **Analytical**: Data-driven, comparison shopping
- **Impatient**: Time pressure, ultimatums
- **Diplomatic**: Professional, win-win focus

**Episode Flow**:
1. Teacher selects persona and scenario
2. RuleAgent initialized with persona config
3. Negotiation proceeds with strict protocol enforcement
4. RuleAgent can terminate on critical violations
5. Episode logged with persona metadata

## SFT Training Pipeline

### Core Component: `training/sft_trainer.py`

**Class: `SFTTrainingPipeline`**

**Initialization**:
```python
pipeline = SFTTrainingPipeline(
    model_name="mistral-7b-instruct",
    dataset_path="data/episodes/sft_packets.jsonl",
    output_dir="models/sft_checkpoint",
    training_params=training_parameters  # From training_parameters.py
)
```

**Key Features**:

1. **QLoRA Configuration**:
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1
)
```

2. **Response Masking** (`utils/response_masking.py`):
- Only response tokens contribute to loss
- Input/instruction tokens masked to -100
- Critical for instruction-following quality

3. **Data Format Support**:
- **Standard SFT**: Simple input/output pairs
- **CraigslistBargain**: Weighted multi-turn format
  - Buyer messages: weight=0 (context only)
  - Seller messages: weight=1 (training target)

4. **Training Configuration**:
```python
training_args = TrainingArguments(
    per_device_train_batch_size=3,
    gradient_accumulation_steps=4,  # Effective batch=12
    learning_rate=5e-5,
    num_train_epochs=0.3,  # For baseline
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100
)
```

5. **Error Handling**:
- Automatic batch size reduction on OOM
- Gradient monitoring for explosions/vanishing
- Loss validation for NaN/inf detection
- Training state recovery from interruptions

### Data Conversion: `training/data_converter.py`

**Episode → SFT Conversion**:
```python
def convert_episodes_to_sft(episodes, output_path):
    """
    Converts ParlAI episodes to SFT format.

    Each turn becomes a training example:
    - Input: state + conversation history + partner message
    - Output: agent's response (no action tags)
    """
    sft_packets = []
    for episode in episodes:
        for i, turn in enumerate(episode['turns']):
            if turn['agent'] == 'seller':  # Or buyer, depending on training
                packet = {
                    'input': build_context(episode, i),
                    'output': turn['message']  # Clean message, no tags
                }
                sft_packets.append(packet)
```

**DPO Pair Generation**:
```python
def auto_generate_dpo(episodes):
    """
    Automatically creates DPO pairs from episodes.

    Uses protocol_validator.calculate_protocol_score() to rank episodes.
    Pairs high-scoring (chosen) with low-scoring (rejected) examples.
    """
    scored = [(ep, calculate_protocol_score(ep)) for ep in episodes]
    sorted_episodes = sorted(scored, key=lambda x: x[1])

    # Top 20% as chosen, bottom 20% as rejected
    chosen_threshold = int(len(sorted_episodes) * 0.8)
    rejected_threshold = int(len(sorted_episodes) * 0.2)
```

## Integration Workflow for Dataset Creator

### 1. Generate Training Data
```python
from training_lab.worlds.selfchat_world import SelfChatWorld
from training_lab.tasks.selfplay_teacher import SelfplayTeacher

# Create world
world = SelfChatWorld(opt, agents, SelfplayTeacher())

# Generate episodes
episodes = []
for batch in range(num_batches):
    batch_episodes = world.run_parallel_episodes(seeds, start_id)
    episodes.extend(batch_episodes)

# Save raw episodes
with open('episodes.jsonl', 'w') as f:
    for ep in episodes:
        f.write(json.dumps(ep) + '\n')
```

### 2. Convert to SFT Format
```python
from training_lab.training.data_converter import convert_episodes_to_sft

# Convert episodes
sft_packets = convert_episodes_to_sft(episodes, 'sft_data.jsonl')
```

### 3. Run SFT Training
```python
from training_lab.training.sft_trainer import SFTTrainingPipeline
from training_lab.training_parameters import *

# Initialize pipeline
pipeline = SFTTrainingPipeline(
    model_name="mistral-7b-instruct",
    dataset_path="sft_data.jsonl",
    output_dir="models/negotiation_sft"
)

# Train
pipeline.train()
```

## Implementation Details

1. **Ignore for now** Optional two-part response flow:
   - OllamaNegotiator generates strategy JSON (internal, 150 tokens max)
   - Then generates public message based on strategy (256 tokens max)
   - Strategy tracked but not included in training loss

2. **Batch Inference in SelfChatWorld**:
   - Groups multiple episodes for parallel GPU processing
   - Maintains separate conversation contexts
   - Significantly faster than sequential generation

3. **Episode Quality Scoring**:
   - `calculate_protocol_score()` evaluates:
     - Deal success, constraint violations, turn efficiency
     - Terms agreement accuracy, message quality
   - Scores used for automatic DPO pair selection

4. **Regression Testing**:
   - 10% of seeds maintained across training iterations
   - Tracks performance degradation on these fixed scenarios
   - Early warning for overfitting or catastrophic forgetting