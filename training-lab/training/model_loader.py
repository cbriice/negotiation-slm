"""Model loading and quantization utilities for QLoRA training."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training
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
    def load_model(model_name: str, device_map: str = None):
        """
        Load model with 4-bit quantization.
        
        Args:
            model_name: HuggingFace model name or local path
            device_map: Device mapping strategy (defaults to params.DEVICE_MAP)
            
        Returns:
            Tuple of (model, tokenizer)
        """
        device_map = device_map or params.DEVICE_MAP
        
        # Load tokenizer
        tokenizer = ModelLoader._setup_tokenizer(model_name)
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=ModelLoader.get_bnb_config(),
            device_map=device_map,
            torch_dtype=params.BNB_4BIT_COMPUTE_DTYPE,
            trust_remote_code=True,
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Enable gradient checkpointing if specified
        if params.GRADIENT_CHECKPOINTING:
            model.gradient_checkpointing_enable()
            
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
        tokenizer.padding_side = "right"
        
        return tokenizer
    
    @staticmethod
    def load_reference_model(model_path: str):
        """
        Load a reference model for DPO training (frozen, 4-bit).
        
        Args:
            model_path: Path to the reference model
            
        Returns:
            Reference model in eval mode
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=ModelLoader.get_bnb_config(),
            device_map=params.DEVICE_MAP,
            torch_dtype=params.BNB_4BIT_COMPUTE_DTYPE,
            trust_remote_code=True,
        )
        
        # Freeze and set to eval mode
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        return model