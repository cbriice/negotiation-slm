"""
Custom evaluation module for negotiation models.

A lightweight alternative to ParlAI for testing and evaluating negotiation models.
"""

from .model_wrapper import SimpleModelWrapper
from .simple_negotiation import NegotiationSession
from .prompt_builder import PromptBuilder

__all__ = ['SimpleModelWrapper', 'NegotiationSession', 'PromptBuilder']