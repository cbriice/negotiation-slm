"""
LLM interface for negotiation dialogue generation using Ollama.

This module provides a robust interface to the Llama-3.3 70B model via Ollama,
with comprehensive error handling, retries, and fallback mechanisms.
"""

import ollama
from typing import Optional, Dict, Any, List
import time
import re
from dataclasses import dataclass

from utils.logger_config import get_logger
from utils.exceptions import (
    LLMGenerationError,
    OllamaConnectionError,
    OllamaModelError
)


@dataclass
class LLMConfig:
    """Configuration for LLM interface."""
    model: str = "llama3.3:70b"
    fallback_model: Optional[str] = "llama3.1:8b"  # Smaller fallback model
    temperature: float = 0.5
    max_retries: int = 3
    retry_backoff: float = 2.0
    timeout: int = 60  # seconds
    max_tokens: int = 256
    use_gpu: bool = True
    gpu_layers: int = 10  # Number of layers to offload to GPU


class LLMInterface:
    """Interface for LLM generation with Ollama."""

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
        self.client = ollama.Client()

        # Verify model availability
        self._verify_models()

        # Track usage statistics
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "fallback_used": 0,
            "total_tokens": 0,
            "total_time": 0
        }

    def _verify_models(self):
        """Verify that required models are available."""
        try:
            models = self.client.list()
            available_models = [m['name'] for m in models.get('models', [])]

            # Check primary model
            if self.config.model not in available_models:
                self.logger.warning(
                    f"Primary model {self.config.model} not found. "
                    f"Available models: {available_models}"
                )
                # Try to pull the model
                self._pull_model(self.config.model)

            # Check fallback model
            if (self.config.fallback_model and
                self.config.fallback_model not in available_models):
                self.logger.warning(
                    f"Fallback model {self.config.fallback_model} not found"
                )

        except Exception as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")

    def _pull_model(self, model_name: str):
        """Attempt to pull a model from Ollama registry."""
        try:
            self.logger.info(f"Pulling model {model_name}...")
            self.client.pull(model_name)
            self.logger.info(f"Successfully pulled {model_name}")
        except Exception as e:
            raise OllamaModelError(f"Failed to pull model {model_name}: {e}")

    def generate(self, prompt: str, temperature: Optional[float] = None,
                sim_id: str = "") -> str:
        """
        Generate response with retry logic and fallback.

        Args:
            prompt: The prompt to generate from
            temperature: Override default temperature
            sim_id: Simulation ID for logging

        Returns:
            Generated response text

        Raises:
            LLMGenerationError: If generation fails after all retries
        """
        start_time = time.time()
        self.stats["total_calls"] += 1

        temp = temperature or self.config.temperature

        # Try with primary model
        try:
            response = self._generate_with_retry(
                prompt, self.config.model, temp, sim_id
            )
            self.stats["successful_calls"] += 1

        except (OllamaConnectionError, OllamaModelError) as e:
            self.logger.warning(
                f"[{sim_id}] Primary model failed: {e}, trying fallback"
            )

            # Try fallback model if available
            if self.config.fallback_model:
                try:
                    response = self._generate_with_retry(
                        prompt, self.config.fallback_model, temp, sim_id
                    )
                    self.stats["fallback_used"] += 1
                    self.stats["successful_calls"] += 1

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

    def _generate_with_retry(self, prompt: str, model: str,
                           temperature: float, sim_id: str) -> str:
        """
        Generate with exponential backoff retry.

        Args:
            prompt: Generation prompt
            model: Model to use
            temperature: Temperature setting
            sim_id: Simulation ID for logging

        Returns:
            Generated text

        Raises:
            OllamaConnectionError: If connection fails
            OllamaModelError: If model errors occur
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                # Make generation request
                response = self.client.generate(
                    model=model,
                    prompt=prompt,
                    options={
                        "temperature": temperature,
                        "num_predict": self.config.max_tokens,
                        "stop": ["\n\n", "BUYER:", "SELLER:"],  # Stop sequences
                        "gpu_layers": self.config.gpu_layers if self.config.use_gpu else 0
                    }
                )

                # Extract and clean response
                generated_text = self._parse_response(response)

                # Validate response
                if not generated_text or len(generated_text.strip()) < 10:
                    raise OllamaModelError("Generated response too short or empty")

                return generated_text

            except ollama.ResponseError as e:
                last_error = e
                self.logger.warning(
                    f"[{sim_id}] Ollama response error (attempt {attempt + 1}): {e}"
                )

                if "out of memory" in str(e).lower():
                    raise OllamaModelError(f"Model out of memory: {e}")

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
        raise OllamaConnectionError(
            f"Failed after {self.config.max_retries} attempts: {last_error}"
        )

    def _parse_response(self, response: Dict[str, Any]) -> str:
        """
        Parse and clean the Ollama response.

        Args:
            response: Raw response from Ollama

        Returns:
            Cleaned response text
        """
        # Get response text
        if isinstance(response, dict):
            text = response.get('response', '')
        else:
            text = str(response)

        # Clean up the response
        text = self._clean_response(text)

        return text

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

    def batch_generate(self, prompts: List[str], temperature: Optional[float] = None,
                      sim_id: str = "") -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of prompts
            temperature: Temperature override
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
            "total_time": 0
        }

    def check_health(self) -> bool:
        """
        Check if LLM service is healthy.

        Returns:
            True if service is accessible and model is loaded
        """
        try:
            # Try to list models
            models = self.client.list()

            # Check if primary model is loaded
            available = [m['name'] for m in models.get('models', [])]
            if self.config.model not in available:
                self.logger.warning(f"Primary model {self.config.model} not loaded")
                return False

            # Try a minimal generation to ensure it works
            test_response = self.client.generate(
                model=self.config.model,
                prompt="Hello",
                options={"num_predict": 5}
            )

            return bool(test_response.get('response'))

        except Exception as e:
            self.error_logger.error(f"Health check failed: {e}")
            return False


# Factory function for creating configured LLM interface
def create_llm_interface(temperature: float = 0.5,
                        use_fallback: bool = True) -> LLMInterface:
    """
    Create a configured LLM interface.

    Args:
        temperature: Default temperature setting
        use_fallback: Whether to use fallback model

    Returns:
        Configured LLMInterface instance
    """
    config = LLMConfig(
        temperature=temperature,
        fallback_model="llama3.1:8b" if use_fallback else None
    )

    return LLMInterface(config)