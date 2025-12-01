"""
Clean HuggingFace model wrapper for negotiation evaluation.
No ParlAI dependencies - just direct model interaction.
"""

import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModelWrapper:
    """Simple wrapper for HuggingFace models with LoRA support."""

    def __init__(self,
                 model_name: str = "mistralai/Mistral-7B-v0.1",
                 adapter_path: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 256,
                 device: Optional[str] = None):
        """
        Initialize model wrapper.

        Args:
            model_name: HuggingFace model ID
            adapter_path: Path to LoRA adapter checkpoint
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            device: Device to use (auto-detected if None)
        """
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model and tokenizer
        self._load_model()

        logger.info(f"Model loaded on {self.device}")

    def _load_model(self):
        """Load the base model and optional LoRA adapter."""
        logger.info(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization for efficient loading
        bnb_config = None
        if self.device.type == 'cuda':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.device.type == 'cuda' else None,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
        )

        # Load LoRA adapter if provided
        if self.adapter_path:
            # Auto-find latest checkpoint if path is a directory
            if Path(self.adapter_path).is_dir():
                checkpoint_path = self._find_latest_checkpoint(self.adapter_path)
                if checkpoint_path:
                    logger.info(f"Loading adapter from: {checkpoint_path}")
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        checkpoint_path,
                        torch_dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32
                    )
                else:
                    logger.warning(f"No checkpoint found in {self.adapter_path}")
            else:
                logger.info(f"Loading adapter from: {self.adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.adapter_path,
                    torch_dtype=torch.bfloat16 if self.device.type == 'cuda' else torch.float32
                )

        # Set model to eval mode
        self.model.eval()

    def _find_latest_checkpoint(self, base_path: str) -> Optional[str]:
        """Find the latest checkpoint in a directory."""
        base_dir = Path(base_path)

        # Look for checkpoint directories
        checkpoint_dirs = sorted([d for d in base_dir.glob("checkpoint-*") if d.is_dir()])

        if checkpoint_dirs:
            # Get the one with highest number
            latest = max(checkpoint_dirs, key=lambda x: int(x.name.split('-')[-1]))
            return str(latest)

        # Check if the base path itself is a valid checkpoint
        if (base_dir / "adapter_model.bin").exists() or (base_dir / "adapter_model.safetensors").exists():
            return str(base_dir)

        return None

    def generate(self,
                 prompt: str,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 stop_sequences: Optional[List[str]] = None) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stop_sequences: List of sequences to stop generation

        Returns:
            Generated text (prompt removed)
        """
        # Use provided values or defaults
        temp = temperature if temperature is not None else self.temperature
        max_new = max_tokens if max_tokens is not None else self.max_tokens

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Set up generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new,
            "temperature": temp,
            "do_sample": temp > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Add top_p for sampling
        if temp > 0:
            gen_kwargs["top_p"] = 0.9

        # Add stop sequences if provided
        if stop_sequences:
            # This is a simplification - proper implementation would need StoppingCriteria
            pass

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )

        # Decode and return only new tokens
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from the output
        if full_output.startswith(prompt):
            response = full_output[len(prompt):].strip()
        else:
            # Sometimes tokenization can cause slight differences
            response = full_output.strip()

        return response

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "model_name": self.model_name,
            "adapter_path": self.adapter_path,
            "device": str(self.device),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Add adapter info if loaded
        if hasattr(self.model, 'peft_config'):
            info["lora_config"] = {
                "r": self.model.peft_config['default'].r,
                "alpha": self.model.peft_config['default'].lora_alpha,
                "dropout": self.model.peft_config['default'].lora_dropout,
            }

        return info