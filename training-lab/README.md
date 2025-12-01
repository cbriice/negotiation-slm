# Training Lab

A utility codebase for training and evaluating negotiation AI models using curriculum learning with weighted masking.

## Quick Start

### Training
```bash
python train_curriculum.py \
  --conversational-data data/training_data/conversational-dataset_formatted.jsonl \
  --negotiation-data data/training_data/negotiation-dataset_formatted.jsonl \
  --output-dir models/curriculum \
  --max-steps 8000
```

### Resume Training
```bash
python resume_curriculum.py \
  --start-from-phase 2 \
  --checkpoint-dir models/curriculum/phase_1_step_2000/
```

### Evaluation
```bash
python evaluation/run_selfchat.py \
  --adapter-path models/curriculum/final \
  --num-seeds 10
```

## Project Structure

```
training-lab/
├── train_curriculum.py          # Main training entry point
├── resume_curriculum.py         # Resume from phase checkpoints
├── curriculum_parameters.py     # Phase definitions & masking weights
├── training_parameters.py       # Global training hyperparameters
│
├── training/                    # Core training infrastructure
│   ├── curriculum/              # Curriculum learning system
│   │   ├── wrapper.py           # Main orchestrator
│   │   ├── masking_collator.py  # Dynamic weighted masking
│   │   ├── data_sampler.py      # Phase-based data mixing
│   │   ├── dynamic_dataset.py   # Streaming dataset
│   │   ├── monitor.py           # Health checks & recovery
│   │   ├── checkpoint_utils.py  # Phase checkpoint management
│   │   └── phase_config.py      # Phase step boundaries
│   ├── model_loader.py          # 4-bit quantization & tokenizer
│   ├── lora_config.py           # LoRA configuration
│   ├── metrics_tracker.py       # Training metrics logging
│   └── converters/              # Data format converters
│
├── evaluation/                  # Model evaluation framework
│   ├── model_wrapper.py         # HuggingFace model wrapper
│   ├── simple_negotiation.py    # Negotiation session manager
│   ├── prompt_builder.py        # Prompt construction
│   ├── eval_seeds.py            # Evaluation seed loader
│   ├── metrics.py               # Evaluation metrics
│   ├── run_selfchat.py          # Self-chat evaluation script
│   └── run_interactive.py       # Interactive testing
│
├── data_processing/             # Data preparation utilities
│   ├── reformat_to_next_utterance.py  # Convert to training format
│   ├── pre_tokenize_datasets.py       # Pre-tokenization
│   ├── analyze_turns.py               # Turn pattern analysis
│   └── analyze_lexical_diversity.py   # Vocabulary analysis
│
├── agents/                      # Agent implementations
│   ├── ollama_wrapper.py        # Ollama LLM integration
│   ├── rule_agent.py            # Rule-based opponent
│   └── token_metrics.py         # Token counting utilities
│
├── config/                      # Configuration modules
│   └── training_config.py       # Dataclass-based config
│
├── schema/                      # Data format definitions
│   ├── rule_agent/              # Rule agent schemas
│   └── new/                     # Updated format examples
│
├── scripts/                     # Utility scripts
│   ├── convert_eval_seeds.py    # Seed format conversion
│   └── test_eval_seeds.py       # Seed validation
│
├── tests/                       # Test suite
├── docs/                        # Documentation
├── data/                        # Training/evaluation data
│
└── _legacy/                     # Deprecated ParlAI-based code
    └── README.md                # Legacy code documentation
```

## Training Pipeline

### Curriculum Learning

The training uses a 3-phase curriculum to prevent catastrophic forgetting when fine-tuning on negotiation tasks:

| Phase | Steps | Data Mix | Focus |
|-------|-------|----------|-------|
| 1. Contextual Learning | 0-2000 | 50% conv, 50% neg | Learn basic patterns |
| 2. Guided Practice | 2000-6000 | 20% conv, 80% neg | Focus on negotiation |
| 3. Specialized Training | 6000-8000 | 5% conv, 95% neg | Pure specialization |

### Weighted Masking

Instead of binary masking, the system uses weighted loss per token:
- **System tokens**: Weight 0.0 (ignored)
- **User tokens**: Weight varies by phase (0.05 → 0.02 → 0.0)
- **Assistant tokens**: Weight 1.0 (full training signal)

This allows the model to learn from context while focusing on generating assistant responses.

### Key Features

- **4-bit Quantization**: BitsAndBytes for memory efficiency
- **LoRA Fine-tuning**: Parameter-efficient training
- **Flash Attention 2**: Faster attention computation
- **Health Monitoring**: Automatic detection of training degeneration
- **Recovery Protocol**: Auto-recovery from critical failures
- **Phase Checkpoints**: Save/resume at phase boundaries

## Evaluation Pipeline

The `evaluation/` module provides a lightweight framework for testing models:

```python
from evaluation import SimpleModelWrapper, NegotiationSession

# Load model
model = SimpleModelWrapper(
    model_name="mistralai/Mistral-7B-v0.1",
    adapter_path="models/curriculum/final"
)

# Run negotiation
session = NegotiationSession(buyer_model=model, seller_model=model)
result = session.run_negotiation(seed_data)
```

### Evaluation Scripts

- **`run_selfchat.py`**: Automated self-play evaluation
- **`run_interactive.py`**: Human vs model testing

## Data Preparation

### 1. Format Conversion
```bash
python data_processing/reformat_to_next_utterance.py \
  --input raw_conversations.jsonl \
  --output formatted_data.jsonl
```

### 2. Pre-tokenization (Optional)
```bash
python data_processing/pre_tokenize_datasets.py \
  --input formatted_data.jsonl \
  --output tokenized_data.jsonl
```

## Configuration

### curriculum_parameters.py
Defines the 3-phase curriculum:
- Phase step boundaries
- Data mixing ratios
- Masking weights per role
- Learning rate multipliers

### training_parameters.py
Global hyperparameters:
- Sequence length, batch size
- LoRA rank and alpha
- Precision settings (bf16, tf32)
- Masking templates

## Legacy Code

The `_legacy/` directory contains deprecated ParlAI-based training infrastructure. This code is preserved for reference but is no longer actively used. See `_legacy/README.md` for details.

**Current approach**: Direct HuggingFace Transformers + TRL
**Deprecated approach**: ParlAI World/Agent abstractions

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers` - Model loading and training
- `trl` - SFTTrainer
- `peft` - LoRA implementation
- `bitsandbytes` - 4-bit quantization
- `accelerate` - Training acceleration

## Documentation

See the `docs/` directory for detailed documentation:
- `CURRICULUM_GUIDE.md` - Quick start guide
- `curriculum_training.md` - Curriculum system details
- `training_theory_explanation.md` - Educational background
- `training_recovery_plan.md` - Debugging guide
