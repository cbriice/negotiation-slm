"""
Custom exceptions for negotiation simulation system.

This module defines specific exception types for different failure modes
to enable targeted error handling and recovery.
"""


class SimulationError(Exception):
    """Base exception for simulation-related errors."""
    pass


class LLMGenerationError(SimulationError):
    """Error during LLM text generation."""
    pass


class ModelLoadError(LLMGenerationError):
    """Failed to load model from Hugging Face."""
    pass


class ModelInferenceError(LLMGenerationError):
    """Error during model inference (OOM, generation failure, etc)."""
    pass


class ControlFunctionError(SimulationError):
    """Error in control function calculation or evaluation."""
    pass


class InvalidStateError(ControlFunctionError):
    """Negotiation state is invalid or inconsistent."""
    pass


class PromptSelectionError(SimulationError):
    """Error selecting appropriate prompt."""
    pass


class InsufficientPromptsError(PromptSelectionError):
    """Not enough unused prompts available."""
    pass


class PersonalityError(SimulationError):
    """Error with personality configuration or initialization."""
    pass


class StateValidationError(SimulationError):
    """State validation failed."""
    pass


class DataValidationError(SimulationError):
    """Data validation failed (schema, coherence, etc)."""
    pass


class SeedLoadError(SimulationError):
    """Error loading or parsing seed data."""
    pass


class CheckpointError(SimulationError):
    """Error with checkpoint save/load operations."""
    pass


class ResourceError(SimulationError):
    """System resource error (disk space, memory, etc)."""
    pass


class ConfigurationError(SimulationError):
    """Configuration or initialization error."""
    pass