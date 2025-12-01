"""
Simple negotiation session manager.
Handles turn-taking and conversation flow without ParlAI complexity.
"""

import re
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

from custom_eval.model_wrapper import SimpleModelWrapper
from custom_eval.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class NegotiationMetrics:
    """Track metrics for a negotiation session."""
    turns: int = 0
    buyer_tokens: int = 0
    seller_tokens: int = 0
    deal_made: bool = False
    final_price: Optional[float] = None
    buyer_started: bool = True
    duration_seconds: float = 0.0
    conversation_quality: float = 0.0  # 0-1 score based on coherence


class NegotiationSession:
    """Manage a negotiation between two model instances."""

    def __init__(self,
                 model_wrapper: SimpleModelWrapper,
                 max_turns: int = 20,
                 temperature: float = 0.7,
                 use_single_model: bool = True,
                 log_file=None):
        """
        Initialize negotiation session.

        Args:
            model_wrapper: Model wrapper to use for generation
            max_turns: Maximum turns in negotiation
            temperature: Generation temperature
            use_single_model: Whether to use the same model for both sides
            log_file: Optional file handle for real-time logging
        """
        self.buyer_model = model_wrapper
        # Use same model for both sides by default (more realistic for self-play)
        self.seller_model = model_wrapper if use_single_model else None

        self.max_turns = max_turns
        self.temperature = temperature
        self.prompt_builder = PromptBuilder(include_strategy_hints=False)
        self.log_file = log_file

        # Conversation state
        self.conversation_history: List[Tuple[str, str]] = []
        self.metrics = NegotiationMetrics()

        # Deal detection patterns
        self.deal_patterns = [
            r"deal",
            r"agreed",
            r"sold",
            r"I'?ll take it",
            r"you'?ve got a deal",
            r"accept.*(?:offer|price)",
            r"works for me",
        ]

        self.rejection_patterns = [
            r"no deal",
            r"can'?t accept",
            r"walk.*away",
            r"not interested",
            r"too (?:high|low|expensive)",
            r"final offer",
        ]

        self.price_pattern = r"\$?(\d+(?:,\d{3})*(?:\.\d{2})?)"

    def run_negotiation(self,
                       seed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete negotiation session.

        Args:
            seed_data: Evaluation seed with item and role information

        Returns:
            Dictionary with conversation history and metrics
        """
        start_time = time.time()

        # Extract data from seed
        item = seed_data.get('item', {})
        buyer_params = seed_data.get('buyer_state', {})
        seller_params = seed_data.get('seller_state', {})

        # Determine who starts (typically buyer)
        current_role = 'buyer'
        self.metrics.buyer_started = True

        # Generate initial prompts with full context
        buyer_context = self.prompt_builder.build_initial_prompt(
            'buyer', item, buyer_params
        )
        seller_context = self.prompt_builder.build_initial_prompt(
            'seller', item, seller_params
        )

        # Track the base context for each side
        contexts = {
            'buyer': buyer_context,
            'seller': seller_context
        }

        # Run negotiation turns
        for turn in range(self.max_turns):
            self.metrics.turns = turn + 1

            # Build prompt with conversation history
            if turn == 0:
                # First turn uses full context
                prompt = contexts[current_role]
            else:
                # Subsequent turns append to context
                prompt = self.prompt_builder.build_initial_prompt(
                    current_role,
                    item,
                    buyer_params if current_role == 'buyer' else seller_params,
                    self.conversation_history
                )

            # Generate response
            try:
                response = self._generate_response(current_role, prompt)

                # Clean response
                response = self.prompt_builder.extract_response(response, current_role)

                # Check for quality issues
                # if self._is_degenerate_response(response):
                #     logger.warning(f"Degenerate response detected at turn {turn}: {response[:100]}")
                #     self.metrics.conversation_quality = max(0, self.metrics.conversation_quality - 0.2)
                #     # Try to recover with a generic response
                #     response = self._get_recovery_response(current_role, turn)
                # else:
                #     self.metrics.conversation_quality = min(1.0, self.metrics.conversation_quality + 0.05)

                self.metrics.conversation_quality = min(1.0, self.metrics.conversation_quality + 0.05)

                # Add to history
                self.conversation_history.append((current_role, response))

                # Log output in real-time
                if self.log_file:
                    self.log_file.write(f"[{current_role.upper()} - Turn {turn + 1}]: {response}\n\n")
                    self.log_file.flush()

                # Update token counts (approximate)
                token_count = len(response.split())
                if current_role == 'buyer':
                    self.metrics.buyer_tokens += token_count
                else:
                    self.metrics.seller_tokens += token_count

                # Check for deal completion
                if self._check_deal_made(response):
                    self.metrics.deal_made = True
                    self.metrics.final_price = self._extract_final_price(response)
                    logger.info(f"Deal made at turn {turn}: ${self.metrics.final_price}")
                    break

                # Check for explicit rejection
                if self._check_rejection(response) and turn > 5:
                    logger.info(f"Negotiation ended with rejection at turn {turn}")
                    break

            except Exception as e:
                logger.error(f"Error generating response at turn {turn}: {e}")
                # Add a fallback response
                response = "I need a moment to think about this offer."
                self.conversation_history.append((current_role, response))

            # Switch roles
            current_role = 'seller' if current_role == 'buyer' else 'buyer'

        # Calculate final metrics
        self.metrics.duration_seconds = time.time() - start_time

        # Build result
        result = {
            'dialogue': self._format_dialogue(),
            'metrics': {
                'turns': self.metrics.turns,
                'deal_made': self.metrics.deal_made,
                'final_price': self.metrics.final_price,
                'buyer_tokens': self.metrics.buyer_tokens,
                'seller_tokens': self.metrics.seller_tokens,
                'total_tokens': self.metrics.buyer_tokens + self.metrics.seller_tokens,
                'duration_seconds': self.metrics.duration_seconds,
                'conversation_quality': self.metrics.conversation_quality,
            },
            'seed_data': seed_data,
            'conversation_history': self.conversation_history,
        }

        return result

    def _generate_response(self, role: str, prompt: str) -> str:
        """Generate response for given role."""
        model = self.buyer_model if role == 'buyer' else (self.seller_model or self.buyer_model)

        response = model.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=150,  # Keep responses concise
            stop_sequences=["\n\n", "User:", "Assistant:", "user:", "assistant:"]
        )

        return response

    # def _is_degenerate_response(self, response: str) -> bool:
    #     """Check if response shows signs of degeneration."""
    #     # Check for repetitive patterns
    #     if len(response) < 10:
    #         return True

    #     # Check for excessive repetition
    #     words = response.split()
    #     if len(words) > 10:
    #         # Check if any word repeats too much
    #         word_counts = {}
    #         for word in words:
    #             word_counts[word] = word_counts.get(word, 0) + 1
    #             # If any word is more than 30% of response, it's degenerate
    #             if word_counts[word] > len(words) * 0.3:
    #                 return True

    #     # Check for nonsense patterns (like "istantistantistant")
    #     if re.search(r'(\w{3,})\1{3,}', response):
    #         return True

    #     # Check for corrupted output patterns
    #     if any(pattern in response.lower() for pattern in ['istant', '''''', '0000', '7\n']):
    #         return True

    #     return False

    # def _get_recovery_response(self, role: str, turn: int) -> str:
    #     """Get a recovery response when generation fails."""
    #     if role == 'buyer':
    #         if turn == 0:
    #             return "Hi, I'm interested in your item. Can you tell me more about it?"
    #         elif turn < 5:
    #             return "That's interesting. What's your best price?"
    #         else:
    #             return "I'll need to think about this offer."
    #     else:  # seller
    #         if turn == 0:
    #             return "Hello! It's a great item in excellent condition."
    #         elif turn < 5:
    #             return "I'm asking a fair price considering the condition and features."
    #         else:
    #             return "What price did you have in mind?"

    def _check_deal_made(self, response: str) -> bool:
        """Check if response indicates a deal was made."""
        response_lower = response.lower()
        return any(re.search(pattern, response_lower) for pattern in self.deal_patterns)

    def _check_rejection(self, response: str) -> bool:
        """Check if response indicates rejection/end of negotiation."""
        response_lower = response.lower()
        return any(re.search(pattern, response_lower) for pattern in self.rejection_patterns)

    def _extract_final_price(self, response: str) -> Optional[float]:
        """Try to extract the final agreed price from response."""
        matches = re.findall(self.price_pattern, response)
        if matches:
            # Return the last mentioned price
            price_str = matches[-1].replace(',', '')
            try:
                return float(price_str)
            except ValueError:
                pass
        return None

    def _format_dialogue(self) -> List[Dict[str, Any]]:
        """Format dialogue for output."""
        dialogue = []
        for i, (speaker, text) in enumerate(self.conversation_history):
            dialogue.append({
                'turn': i,
                'speaker': speaker,
                'text': text,
                'tokens': len(text.split()),  # Approximate
            })
        return dialogue

    def reset(self):
        """Reset session for new negotiation."""
        self.conversation_history = []
        self.metrics = NegotiationMetrics()