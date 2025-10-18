"""
LLM interface for negotiation dialogue generation using Hugging Face transformers.

This module provides a robust interface to the Qwen2.5-14B-Instruct model via transformers,
with comprehensive error handling, retries, and fallback mechanisms.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any, List
import time
import re
from dataclasses import dataclass
import gc

from utils.logger_config import get_logger
from utils.exceptions import (
    LLMGenerationError,
    ModelLoadError,
    ModelInferenceError
)


@dataclass
class LLMConfig:
    """Configuration for LLM interface."""
    model: str = "Qwen/Qwen2.5-14B-Instruct"
    fallback_model: Optional[str] = "Qwen/Qwen2.5-7B-Instruct"  # Smaller fallback model
    temperature: Optional[float] = None  # Temperature should be provided per generation from personality
    max_retries: int = 3
    retry_backoff: float = 2.0
    timeout: int = 60  # seconds
    max_tokens: int = 256
    use_gpu: bool = True
    device_map: str = "auto"  # Auto device mapping for multi-GPU/CPU
    load_in_4bit: bool = True  # Use 4-bit quantization to save memory
    torch_dtype: str = "auto"  # Data type for model weights


class LLMInterface:
    """Interface for LLM generation with Hugging Face transformers."""

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM interface.

        Args:
            config: LLM configuration, uses defaults if None
        """
        self.config = config or LLMConfig()
        # Get the SimulationLogger instance
        self.sim_logger_instance = get_logger()
        # Get individual loggers for different logging needs
        self.logger = self.sim_logger_instance.sim_logger  # Main logger for warnings/info
        self.llm_logger = self.sim_logger_instance.llm_logger  # For LLM-specific logging
        self.error_logger = self.sim_logger_instance.error_logger  # For errors

        # Initialize models and tokenizers
        self.model = None
        self.tokenizer = None
        self.fallback_model = None
        self.fallback_tokenizer = None

        # Device management
        self.device = "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"

        # Load primary model
        self._load_primary_model()

        # Track usage statistics
        self.stats: Dict[str, float] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "fallback_used": 0,
            "total_tokens": 0,
            "total_time": 0.0
        }

    def _load_primary_model(self):
        """Load the primary model and tokenizer."""
        try:
            self.logger.info(f"Loading model {self.config.model}...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                trust_remote_code=True
            )

            # Set pad token if not set (some models don't have it)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Configure model loading kwargs
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": self.config.device_map if self.config.use_gpu else None
            }

            # Add quantization config if using 4-bit
            if self.config.load_in_4bit and self.config.use_gpu:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                # Don't pass load_in_4bit separately when using quantization_config

            # Set dtype (torch_dtype is deprecated)
            if self.config.torch_dtype == "auto":
                model_kwargs["dtype"] = "auto"
            else:
                model_kwargs["dtype"] = getattr(torch, self.config.torch_dtype, torch.float16)

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                **model_kwargs
            )

            self.logger.info(f"Successfully loaded {self.config.model}")

        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.config.model}: {e}")

    def _load_fallback_model(self):
        """Load the fallback model if not already loaded."""
        if self.fallback_model is not None:
            return  # Already loaded

        if not self.config.fallback_model:
            return  # No fallback configured

        try:
            self.logger.info(f"Loading fallback model {self.config.fallback_model}...")

            # Load tokenizer
            self.fallback_tokenizer = AutoTokenizer.from_pretrained(
                self.config.fallback_model,
                trust_remote_code=True
            )

            # Set pad token if not set (some models don't have it)
            if self.fallback_tokenizer.pad_token is None:
                self.fallback_tokenizer.pad_token = self.fallback_tokenizer.eos_token

            # Configure model loading kwargs for smaller fallback model
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": self.config.device_map if self.config.use_gpu else None,
                "dtype": torch.float16 if self.config.use_gpu else torch.float32
            }

            # Fallback model can use 4-bit too for memory efficiency
            if self.config.load_in_4bit and self.config.use_gpu:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                # Don't pass load_in_4bit separately when using quantization_config

            # Load model
            self.fallback_model = AutoModelForCausalLM.from_pretrained(
                self.config.fallback_model,
                **model_kwargs
            )

            self.logger.info(f"Successfully loaded fallback {self.config.fallback_model}")

        except Exception as e:
            self.logger.warning(f"Failed to load fallback model: {e}")

    def generate(self, prompt: str, temperature: float,
                sim_id: str = "") -> str:
        """
        Generate response with retry logic and fallback.

        Args:
            prompt: The prompt to generate from
            temperature: Temperature for generation (from personality traits)
            sim_id: Simulation ID for logging

        Returns:
            Generated response text

        Raises:
            LLMGenerationError: If generation fails after all retries
        """
        start_time = time.time()
        self.stats["total_calls"] += 1

        # Temperature is now required and must come from personality
        temp = temperature

        # Try with primary model
        try:
            response = self._generate_with_retry(
                prompt, self.model, self.tokenizer, temp, sim_id, is_primary=True
            )
            self.stats["successful_calls"] += 1

        except Exception as e:
            self.logger.warning(
                f"[{sim_id}] Primary model failed: {e}, trying fallback"
            )

            # Try fallback model if available
            if self.config.fallback_model:
                try:
                    # Load fallback model if not already loaded
                    self._load_fallback_model()

                    if self.fallback_model and self.fallback_tokenizer:
                        response = self._generate_with_retry(
                            prompt, self.fallback_model, self.fallback_tokenizer,
                            temp, sim_id, is_primary=False
                        )
                        self.stats["fallback_used"] += 1
                        self.stats["successful_calls"] += 1
                    else:
                        raise LLMGenerationError("Fallback model failed to load")

                except Exception as fallback_error:
                    self.stats["failed_calls"] += 1
                    raise LLMGenerationError(
                        f"Both primary and fallback models failed: "
                        f"Primary: {e}, Fallback: {fallback_error}"
                    )
            else:
                self.stats["failed_calls"] += 1
                raise

        # Log performance
        duration = time.time() - start_time
        self.stats["total_time"] += duration
        self.sim_logger_instance.log_llm_call(sim_id, prompt, response, duration, True)

        return response

    def _generate_with_retry(self, prompt: str, model, tokenizer,
                           temperature: float, sim_id: str, is_primary: bool = True) -> str:
        """
        Generate with exponential backoff retry using Hugging Face transformers.

        Args:
            prompt: Generation prompt
            model: Model instance to use
            tokenizer: Tokenizer instance to use
            temperature: Temperature setting
            sim_id: Simulation ID for logging
            is_primary: Whether this is the primary model

        Returns:
            Generated text

        Raises:
            LLMGenerationError: If generation fails
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                # Tokenize the input
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=2048)

                # Move inputs to model device if using GPU
                if self.config.use_gpu and hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Generate with model
                with torch.no_grad():
                    # Base generation kwargs
                    gen_kwargs = {
                        **inputs,
                        "max_new_tokens": self.config.max_tokens,
                        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                        "eos_token_id": tokenizer.eos_token_id
                    }

                    # Only add sampling parameters when actually sampling
                    if temperature > 0:
                        gen_kwargs.update({
                            "do_sample": True,
                            "temperature": temperature,
                            "top_p": 0.95
                        })
                    else:
                        gen_kwargs["do_sample"] = False

                    outputs = model.generate(**gen_kwargs)

                # Decode the output
                generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Clean the response
                generated_text = self._clean_response(generated_text)

                # Check for degenerate response (excessive newlines, vertical text, etc.)
                if self._is_degenerate_response(generated_text):
                    raise ModelInferenceError("Generated response is degenerate (excessive newlines or repetitive pattern)")

                # Validate response
                if not generated_text or len(generated_text.strip()) < 10:
                    raise ModelInferenceError("Generated response too short or empty")

                # Track token usage
                self.stats["total_tokens"] += len(generated_ids)

                return generated_text

            except torch.cuda.OutOfMemoryError as e:
                last_error = e
                self.logger.warning(
                    f"[{sim_id}] GPU OOM error (attempt {attempt + 1}): {e}"
                )
                # Try to free up memory
                if self.config.use_gpu:
                    torch.cuda.empty_cache()
                    gc.collect()

                    # If OOM on primary, might need to offload more to CPU
                    if is_primary and attempt == 0:
                        self.logger.info("Attempting to clear GPU cache and retry...")

            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"[{sim_id}] Generation error (attempt {attempt + 1}): {e}"
                )

            # Exponential backoff
            if attempt < self.config.max_retries - 1:
                wait_time = self.config.retry_backoff ** attempt
                self.llm_logger.debug(f"[{sim_id}] Retrying in {wait_time}s...")
                time.sleep(wait_time)

        # All retries failed
        raise LLMGenerationError(
            f"Failed after {self.config.max_retries} attempts: {last_error}"
        )


    def _is_degenerate_response(self, text: str) -> bool:
        """
        Check if response is degenerate (e.g., excessive newlines, repetitive characters).

        Args:
            text: Text to check

        Returns:
            True if text is degenerate, False otherwise
        """
        if not text:
            return True

        # Check for excessive newline density (>30% of characters are newlines)
        newline_ratio = text.count('\n') / max(len(text), 1)
        if newline_ratio > 0.3:
            return True

        # Check for vertical text pattern (single characters separated by newlines)
        lines = text.split('\n')
        if len(lines) > 5:
            single_char_lines = sum(1 for line in lines if len(line.strip()) <= 1)
            if single_char_lines / len(lines) > 0.5:
                return True

        # Check for extreme repetition
        if len(text) > 20:
            unique_chars = len(set(text.replace('\n', '').replace(' ', '')))
            if unique_chars < 5:  # Very few unique characters
                return True

        return False

    def _clean_response(self, text: str) -> str:
        """
        Clean up generated response text.

        Args:
            text: Raw generated text

        Returns:
            Cleaned text
        """
        # Remove any system-like tokens or meta-text
        patterns_to_remove = [
            r'\[.*?\]',  # Remove bracketed content
            r'<.*?>',    # Remove HTML-like tags
            r'^\s*\n',   # Remove leading newlines
            r'\n\s*$',   # Remove trailing newlines
            r'Response:?\s*',  # Remove "Response:" prefix
            r'Output:?\s*',    # Remove "Output:" prefix
        ]

        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Ensure single response (no multi-turn generation)
        if '\nBUYER:' in text or '\nSELLER:' in text:
            # Cut off at first role marker
            for marker in ['\nBUYER:', '\nSELLER:']:
                if marker in text:
                    text = text.split(marker)[0]

        # Trim whitespace
        text = text.strip()

        # Ensure reasonable length
        if len(text) > 500:
            # Find last complete sentence
            sentences = text.split('.')
            if len(sentences) > 1:
                text = '.'.join(sentences[:-1]) + '.'

        return text

    def batch_generate(self, prompts: List[str], temperature: float,
                      sim_id: str = "") -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of prompts
            temperature: Temperature for generation (from personality traits)
            sim_id: Simulation ID

        Returns:
            List of generated responses
        """
        responses = []

        for i, prompt in enumerate(prompts):
            try:
                response = self.generate(prompt, temperature, f"{sim_id}_batch_{i}")
                responses.append(response)
            except LLMGenerationError as e:
                self.error_logger.error(f"[{sim_id}] Batch generation failed at index {i}: {e}")
                responses.append("")  # Add empty response to maintain alignment

        return responses

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        stats = self.stats.copy()

        # Calculate averages
        if stats["successful_calls"] > 0:
            stats["avg_time_per_call"] = stats["total_time"] / stats["successful_calls"]
        else:
            stats["avg_time_per_call"] = 0

        if stats["total_calls"] > 0:
            stats["success_rate"] = stats["successful_calls"] / stats["total_calls"]
            stats["fallback_rate"] = stats["fallback_used"] / stats["total_calls"]
        else:
            stats["success_rate"] = 0
            stats["fallback_rate"] = 0

        return stats

    def reset_stats(self):
        """Reset usage statistics."""
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "fallback_used": 0,
            "total_tokens": 0,
            "total_time": 0.0
        }

    def check_health(self) -> bool:
        """
        Check if LLM service is healthy.

        Returns:
            True if service is accessible and model is loaded
        """
        try:
            # Check if primary model and tokenizer are loaded
            if self.model is None or self.tokenizer is None:
                self.logger.warning("Primary model or tokenizer not loaded")
                return False

            # Try a minimal generation to ensure it works
            test_prompt = "Hello, this is a test."
            inputs = self.tokenizer(test_prompt, return_tensors="pt", truncation=True)

            # Move to device if using GPU
            if self.config.use_gpu and hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Try to generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False
                )

            # Check if we got valid output
            return outputs is not None and len(outputs[0]) > len(inputs['input_ids'][0])

        except Exception as e:
            self.error_logger.error(f"Health check failed: {e}")
            return False

    def cleanup(self):
        """Clean up resources and free memory."""
        try:
            # Clear GPU cache if using GPU
            if self.config.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Delete models to free memory
            if self.model is not None:
                del self.model
                self.model = None

            if self.fallback_model is not None:
                del self.fallback_model
                self.fallback_model = None

            # Run garbage collection
            gc.collect()

            self.logger.info("LLM interface cleaned up successfully")

        except Exception as e:
            self.error_logger.error(f"Error during cleanup: {e}")


# Factory function for creating configured LLM interface
def create_llm_interface(use_fallback: bool = True,
                        load_in_4bit: bool = True) -> LLMInterface:
    """
    Create a configured LLM interface.

    Note: Temperature should be provided per generation call from personality traits.

    Args:
        use_fallback: Whether to use fallback model
        load_in_4bit: Whether to use 4-bit quantization

    Returns:
        Configured LLMInterface instance
    """
    config = LLMConfig(
        fallback_model="Qwen/Qwen2.5-7B-Instruct" if use_fallback else None,
        load_in_4bit=load_in_4bit
    )

    return LLMInterface(config)