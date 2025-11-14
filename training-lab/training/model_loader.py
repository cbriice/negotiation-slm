"""Model loading and quantization utilities for QLoRA training."""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft.utils.other import prepare_model_for_kbit_training
import training_parameters as params


class ModelLoader:
    """Handles model loading with 4-bit quantization for QLoRA training."""
    
    @staticmethod
    def get_bnb_config() -> BitsAndBytesConfig:
        """Create BitsAndBytes configuration for 4-bit quantization."""
        return BitsAndBytesConfig(
            load_in_4bit=params.LOAD_IN_4BIT,
            bnb_4bit_quant_type=params.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=params.BNB_4BIT_COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=params.BNB_4BIT_USE_DOUBLE_QUANT,
        )
    
    @staticmethod
    def load_model(model_name: str, device_map: str | None = None, use_flash_attention_2: bool = True):
        """
        Load model with 4-bit quantization and Flash Attention 2.

        Args:
            model_name: HuggingFace model name or local path
            device_map: Device mapping strategy (defaults to params.DEVICE_MAP)
            use_flash_attention_2: Whether to use Flash Attention 2 (default: True)

        Returns:
            Tuple of (model, tokenizer)
        """
        device_map = device_map or params.DEVICE_MAP
        
        # Load tokenizer
        tokenizer = ModelLoader._setup_tokenizer(model_name)
        
        # Load model with quantization and Flash Attention 2
        model_kwargs = {
            "quantization_config": ModelLoader.get_bnb_config(),
            "device_map": device_map,
            "torch_dtype": params.BNB_4BIT_COMPUTE_DTYPE,
            "trust_remote_code": True,
        }

        # Enable Flash Attention 2 if supported
        if use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=params.GRADIENT_CHECKPOINTING
        )
        
        # Enable gradient checkpointing if specified
        if params.GRADIENT_CHECKPOINTING:
            # Use use_reentrant=False for better memory efficiency and to avoid deprecation warning
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            
        return model, tokenizer
    
    @staticmethod
    def _setup_tokenizer(model_name: str):
        """Setup tokenizer with proper padding configuration."""
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Set padding token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        # Set padding side for training
        # Use left padding for Flash Attention compatibility with Mistral
        tokenizer.padding_side = "left"
        
        return tokenizer
    
    @staticmethod
    def load_reference_model(model_path: str, use_flash_attention_2: bool = True):
        """
        Load a reference model for DPO training (frozen, 4-bit).

        Args:
            model_path: Path to the reference model
            use_flash_attention_2: Whether to use Flash Attention 2 (default: True)

        Returns:
            Reference model in eval mode
        """
        model_kwargs = {
            "quantization_config": ModelLoader.get_bnb_config(),
            "device_map": params.DEVICE_MAP,
            "torch_dtype": params.BNB_4BIT_COMPUTE_DTYPE,
            "trust_remote_code": True,
        }

        # Enable Flash Attention 2 if supported
        if use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        # Freeze and set to eval mode
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        return model