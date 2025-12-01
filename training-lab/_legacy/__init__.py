"""
Legacy code - ParlAI-based training and evaluation infrastructure.

This code is preserved but no longer actively used. The main training pipeline
has moved to train_curriculum.py with the training/curriculum/ module.

The custom evaluation pipeline in evaluation/ (formerly custom_eval/) is the
preferred evaluation approach.

Contents:
- tasks/: ParlAI FixedDialogTeacher subclasses for data loading
- training/: Legacy trainers (SFT, DPO, QLoRA) that depend on ParlAI
- evaluation/: ParlAI-integrated metrics system
- utils/: Utility modules used by legacy trainers
- scripts/: ParlAI runner scripts (reference deleted worlds)
- agents/: ParlAI Agent base class
- parlai_config.py: ParlAI framework configuration
"""
