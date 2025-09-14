"""LoRA adapter configuration for PEFT."""

import logging
from peft import LoraConfig, get_peft_model, TaskType
import training_parameters as params

# Set up logging
logger = logging.getLogger(__name__)


class LoRAConfigurator:
    """Manages LoRA adapter configuration and initialization."""
    
    @staticmethod
    def get_lora_config() -> LoraConfig:
        """Create LoRA configuration from training parameters."""
        logger.info("Creating LoRA configuration")
        logger.debug(f"LoRA parameters: r={params.R}, alpha={params.ALPHA}, dropout={params.DROPOUT}")
        logger.debug(f"Target modules: {params.TARGET_MODULES}")
        
        config = LoraConfig(
            r=params.R,
            lora_alpha=params.ALPHA,
            lora_dropout=params.DROPOUT,
            target_modules=params.TARGET_MODULES,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        logger.info(f"LoRA config created successfully with r={params.R}")
        return config
    
    @staticmethod
    def apply_lora(model, lora_config: LoraConfig = None):
        """
        Apply LoRA adapters to model.
        
        Args:
            model: The base model to apply LoRA to
            lora_config: Optional custom LoRA config (defaults to params config)
            
        Returns:
            Model with LoRA adapters applied
        """
        if lora_config is None:
            lora_config = LoRAConfigurator.get_lora_config()
        
        logger.info("Applying LoRA adapters to model")
        logger.debug(f"Model type: {type(model).__name__}")
        
        # Apply PEFT
        try:
            model = get_peft_model(model, lora_config)
            logger.info("LoRA adapters applied successfully")
        except Exception as e:
            logger.error(f"Failed to apply LoRA adapters: {e}")
            raise
        
        # Print trainable parameters info
        LoRAConfigurator._print_trainable_parameters(model)
        
        return model
    
    @staticmethod
    def _print_trainable_parameters(model):
        """Print number of trainable parameters."""
        logger.info("Calculating trainable parameters")
        trainable_params = 0
        all_params = 0
        
        for name, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                logger.debug(f"Trainable layer: {name} - {param.numel():,} params")
                
        if all_params > 0:
            trainable_pct = 100 * trainable_params / all_params
            param_info = (f"Trainable params: {trainable_params:,} || "
                         f"All params: {all_params:,} || "
                         f"Trainable %: {trainable_pct:.2f}")
            print(param_info)  # Keep print for visibility
            logger.info(param_info)
        else:
            param_info = (f"Trainable params: {trainable_params:,} || "
                         f"All params: {all_params:,}")
            print(param_info)  # Keep print for visibility
            logger.warning(f"No parameters found in model: {param_info}")
        
    @staticmethod
    def merge_and_unload(model):
        """
        Merge LoRA weights back into base model and unload adapters.
        
        Args:
            model: PEFT model with LoRA adapters
            
        Returns:
            Base model with merged weights
        """
        logger.info("Merging LoRA weights back into base model")
        try:
            merged_model = model.merge_and_unload()
            logger.info("Successfully merged and unloaded LoRA adapters")
            return merged_model
        except Exception as e:
            logger.error(f"Failed to merge and unload LoRA adapters: {e}")
            raise