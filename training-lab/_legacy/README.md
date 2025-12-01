# Legacy Code Archive

This directory contains deprecated ParlAI-based training and evaluation infrastructure.

## Why This Code Was Deprecated

The ParlAI framework was abandoned due to:
1. Complex integration overhead with the World/Agent abstraction
2. Difficulty debugging within the ParlAI workflow
3. Inflexibility for custom training approaches (curriculum learning)
4. The framework had "weird funks" that made it difficult to work with

## Current Active Pipelines

### Training
- **Entry point**: `train_curriculum.py` (root)
- **Core module**: `training/curriculum/`
- Uses direct HuggingFace Transformers + TRL without ParlAI

### Evaluation
- **Module**: `evaluation/` (formerly `custom_eval/`)
- Lightweight, ParlAI-independent evaluation framework

## Contents

### tasks/
ParlAI `FixedDialogTeacher` subclasses:
- `negotiation_teacher.py` - Base teacher for JSONL episodes
- `selfplay_teacher.py` - Self-play scenario teacher
- `rule_opponent_teacher.py` - Rule-based opponent teacher
- `human_eval_teacher.py` - Human evaluation teacher

### training/
Legacy trainers:
- `sft_trainer.py` - Supervised Fine-Tuning
- `dpo_trainer.py` - Direct Preference Optimization
- `qlora_trainer.py` - QLoRA trainer
- `weighted_sft_trainer.py` - Weighted loss SFT
- `checkpoint_manager.py` - Checkpoint management
- `data_converter.py` - Data format conversion

### evaluation/
ParlAI-integrated metrics:
- `episode_metrics.py` - Per-episode metrics
- `session_metrics.py` - Session aggregation
- `csv_logger.py` - CSV persistence
- `metrics_integration.py` - ParlAI bridge

### utils/
Utility modules for legacy trainers:
- Dataset loading, collators, monitors, validators, etc.

### scripts/
Broken ParlAI runner scripts (reference deleted World classes):
- `run_selfchat.py` - SelfChatWorld runner
- `run_dialog.py` - DialogPartnerWorld runner
- `train.py` - Legacy training orchestrator

### agents/
- `negotiator_base.py` - ParlAI Agent subclass with protocol validation

### parlai_config.py
Central ParlAI configuration dictionary.

## Deleted Code (Git History)

The following World implementations were deleted in commit `1ddfbcb7`:
- `worlds/selfchat_world.py` - Self-play simulation
- `worlds/dialog_world.py` - Dialog partner training
- `worlds/dialogue_flow.py` - Dialogue flow management
- `worlds/batch_inference.py` - GPU batch inference

These can be recovered from git history if needed.

## Import Paths

**Note**: The code within `_legacy/` still uses old import paths (e.g., `from utils.X` instead of `from _legacy.utils.X`). This is intentional - the code is preserved as-is for reference. If you need to run any of this legacy code, you would need to update the import paths.

Code outside of `_legacy/` that references legacy utilities (like `agents/rule_agent.py`) has been updated to use `_legacy.` prefix imports.
