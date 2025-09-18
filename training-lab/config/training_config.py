import sys
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

# Import existing training parameters
sys.path.insert(0, str(Path(__file__).parent.parent))
from training_parameters import *

@dataclass
class TrainingConfig:
    """Configuration for different training stages."""
    stage: str
    model_name: str = "mistralai/Mistral-7B-v0.1"  # HuggingFace model name
    output_dir: str = "./checkpoints"
    adapter_path: Optional[str] = None  # Path to LoRA adapters
    use_latest_checkpoint: bool = True  # Auto-load latest checkpoint
    
    # Training hyperparameters (inherit from training_parameters.py)
    max_seq_len: int = MAX_SEQ_LEN
    per_device_batch_size: int = PER_DEVICE_BATCH_SIZE
    gradient_accumulation_steps: int = GRAD_ACCUM_STEPS
    seed: int = SEED
    
    # QLoRA settings
    load_in_4bit: bool = LOAD_IN_4BIT
    bnb_4bit_quant_type: str = BNB_4BIT_QUANT_TYPE
    bnb_4bit_compute_dtype: Any = BNB_4BIT_COMPUTE_DTYPE
    bnb_4bit_use_double_quant: bool = BNB_4BIT_USE_DOUBLE_QUANT
    
    # LoRA settings
    lora_r: int = R
    lora_alpha: int = ALPHA
    lora_dropout: float = DROPOUT
    target_modules: list = []
    
    # Optimizer settings
    gradient_checkpointing: bool = GRADIENT_CHECKPOINTING
    optim: str = OPTIM
    bf16: bool = BF16
    device_map: str = DEVICE_MAP
    packing: bool = PACKING
    
    # Logging and saving
    logging_steps: int = LOGGING_STEPS
    save_steps: int = SAVE_STEPS
    save_strategy: str = SAVE_STRATEGY
    eval_strategy: str = EVAL_STRATEGY
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = TARGET_MODULES.copy()
        
        # Create output directory
        output_path = Path(self.output_dir) / self.stage
        output_path.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(output_path)

# Stage-specific configurations
STAGE_CONFIGS = {
    'baseline': {
        'description': 'Initial baseline evaluation - no training',
        'max_steps': 0,  # No training, just evaluation
        'learning_rate': 0,
        'eval_only': True,
    },
    'warmstart': {
        'description': 'Warm-start with Deal-or-No-Deal and Casino tasks',
        'max_steps': 500,  # ~0.5 epoch each task
        'learning_rate': SFT_LR,
        'warmup_ratio': SFT_WARMUP_RATIO,
        'lr_scheduler_type': SFT_LR_SCHEDULER,
        'save_as_reference': 'negotiation_baseline',  # For DPO reference
    },
    'selfchat': {
        'description': 'Self-chat training with ~1500 episodes',
        'max_steps': SFT_MAX_STEPS,
        'learning_rate': SFT_LR,
        'warmup_ratio': SFT_WARMUP_RATIO,
        'lr_scheduler_type': SFT_LR_SCHEDULER,
    },
    'dpo_selfchat': {
        'description': 'DPO training on self-chat good/bad pairs',
        'max_steps': DPO_MAX_STEPS,
        'learning_rate': DPO_LR,
        'beta': DPO_BETA,
        'label_smoothing': DPO_LABEL_SMOOTH,
        'max_prompt_length': DPO_MAX_PROMPT_LEN,
        'trainer_type': 'dpo',
    },
    'dialogpartner': {
        'description': 'Training with RuleAgent opponents',
        'max_steps': SFT_MAX_STEPS,
        'learning_rate': SFT_LR,
        'warmup_ratio': SFT_WARMUP_RATIO,
        'lr_scheduler_type': SFT_LR_SCHEDULER,
    },
    'dpo_dialogpartner': {
        'description': 'DPO training on DialogPartner good/bad pairs',
        'max_steps': DPO_MAX_STEPS,
        'learning_rate': DPO_LR,
        'beta': DPO_BETA,
        'label_smoothing': DPO_LABEL_SMOOTH,
        'max_prompt_length': DPO_MAX_PROMPT_LEN,
        'trainer_type': 'dpo',
    },
    'micro_rft': {
        'description': 'Micro reinforcement fine-tuning on top examples',
        'max_steps': 100,  # Very short, 0.1-0.3 epoch
        'learning_rate': SFT_LR * 0.1,  # Lower LR for fine adjustments
        'warmup_ratio': 0,
        'lr_scheduler_type': 'constant',
    },
    'augmented': {
        'description': 'Training with augmented/mutated data',
        'max_steps': SFT_MAX_STEPS,
        'learning_rate': SFT_LR,
        'warmup_ratio': SFT_WARMUP_RATIO,
        'lr_scheduler_type': SFT_LR_SCHEDULER,
        'use_augmentation': True,
    },
}

def get_stage_config(stage: str, **overrides) -> TrainingConfig:
    """Get training configuration for a specific stage."""
    if stage not in STAGE_CONFIGS:
        raise ValueError(f"Unknown stage: {stage}. Available: {list(STAGE_CONFIGS.keys())}")
    
    config = TrainingConfig(stage=stage)
    stage_settings = STAGE_CONFIGS[stage]
    
    # Apply stage-specific settings
    for key, value in stage_settings.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Apply any overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

def get_checkpoint_path(stage: str, step: Optional[int] = None) -> Path:
    """Get the checkpoint path for a given stage and step."""
    base_path = Path('./checkpoints') / stage
    if step is not None:
        return base_path / f'checkpoint-{step}'
    else:
        # Return the latest checkpoint
        checkpoints = sorted(base_path.glob('checkpoint-*'))
        return checkpoints[-1] if checkpoints else base_path

def get_latest_adapter_path() -> Optional[Path]:
    """Find the most recent adapter checkpoint across all stages."""
    checkpoints_dir = Path('./checkpoints')
    if not checkpoints_dir.exists():
        return None

    # Find all adapter_model.bin files with their modification times
    adapter_files = []
    for stage_dir in checkpoints_dir.iterdir():
        if stage_dir.is_dir():
            for checkpoint_dir in stage_dir.glob('checkpoint-*'):
                adapter_file = checkpoint_dir / 'adapter_model.bin'
                if adapter_file.exists():
                    adapter_files.append((adapter_file.stat().st_mtime, checkpoint_dir))

    if adapter_files:
        # Return the checkpoint directory with the most recent adapter
        adapter_files.sort(reverse=True)
        return adapter_files[0][1]

    return None