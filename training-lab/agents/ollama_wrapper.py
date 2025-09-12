import logging
import json
import time
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import requests

from agents.negotiator_base import NegotiatorBase, NegotiationAction
from agents.token_metrics import TokenMetrics
from training_parameters import (
    USE_TWO_PART_RESPONSE, STRATEGY_MAX_TOKENS, RESPONSE_MAX_TOKENS,
    STRATEGY_TEMPERATURE, MODEL_TEMP
)
from worlds.dialogue_flow import DialogueFlowManager

# Set up logging for agent-related logs
Path('logs').mkdir(exist_ok=True)

# Configure logging with 'w' mode to clear log on each run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/agent_logs.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

class OllamaNegotiator(NegotiatorBase):
    """Ollama model wrapped as a ParlAI negotiation agent."""
    
    @classmethod
    def add_cmdline_args(cls, argparser, partial_opt=None):
        """Add command-line arguments specific to this agent."""
        agent = argparser.add_argument_group('Ollama Negotiator Arguments')
        agent.add_argument(
            '--model-name',
            type=str,
            default='mistral:7b',
            help='Ollama model name (e.g., mistral:7b, mistral:latest)'
        )
        agent.add_argument(
            '--ollama-host',
            type=str,
            default='http://localhost:11434',
            help='Ollama API host URL'
        )
        agent.add_argument(
            '--temperature',
            type=float,
            default=MODEL_TEMP,  # Use 0.5 from training_parameters.py
            help='Temperature for generation'
        )
        agent.add_argument(
            '--top-p',
            type=float,
            default=0.9,
            help='Top-p for nucleus sampling'
        )
        agent.add_argument(
            '--max-tokens',
            type=int,
            default=256,
            help='Maximum tokens to generate'
        )
        agent.add_argument(
            '--timeout',
            type=int,
            default=30,
            help='Request timeout in seconds'
        )
        agent.add_argument(
            '--system-prompt',
            type=str,
            default=None,
            help='System prompt for the model'
        )
        return argparser
    
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        
        # Model configuration
        self.model_name = opt.get('model_name', 'mistral:7b')
        self.ollama_host = opt.get('ollama_host', 'http://localhost:11434')
        self.temperature = opt.get('temperature', MODEL_TEMP)
        self.top_p = opt.get('top_p', 0.9)
        self.max_tokens = opt.get('max_tokens', 256)
        self.timeout = opt.get('timeout', 30)
        self.system_prompt = opt.get('system_prompt', None)
        
        # Two-part response configuration
        self.use_two_part_response = opt.get('use_two_part_response', USE_TWO_PART_RESPONSE)
        self.strategy_temperature = opt.get('strategy_temperature', STRATEGY_TEMPERATURE)
        self.strategy_max_tokens = opt.get('strategy_max_tokens', STRATEGY_MAX_TOKENS)
        self.response_max_tokens = opt.get('response_max_tokens', RESPONSE_MAX_TOKENS)
        
        # Enhanced token tracking
        self.token_metrics = TokenMetrics()
        
        # Dialogue flow manager for prompt generation
        self.dialogue_flow = DialogueFlowManager()
        
        # Test connection to Ollama
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama service."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if self.model_name not in model_names:
                    logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {model_names}")
                else:
                    logger.info(f"Successfully connected to Ollama. Using model: {self.model_name}")
            else:
                logger.error(f"Failed to connect to Ollama: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama at {self.ollama_host}: {e}")
            logger.info("Make sure Ollama is running: 'ollama serve'")
    
    def _build_strategy_prompt(self) -> str:
        """Build the prompt for strategy generation."""
        prompt_parts = []
        
        # Role and context
        side = self.side if self.side else "negotiator"
        prompt_parts.append(f"You are a {side} developing a negotiation strategy to gain an advantage.")
        prompt_parts.append("")
        
        # Current state
        prompt_parts.append("Current State:")
        if self.current_state and 'item' in self.current_state:
            item = self.current_state['item']
            prompt_parts.append(f"- Item: {item.get('name', 'Unknown')}")
            prompt_parts.append(f"- Condition: {item.get('condition', 'Unknown')}")
            prompt_parts.append(f"- List Price: ${item.get('list_price', 0)}")
        
        # Role constraints
        if self.role_state:
            if self.side == 'buyer':
                max_price = self.role_state.get('max_price', 0)
                target_price = self.role_state.get('target_price', 0)
                prompt_parts.append(f"- Your maximum price: ${max_price}")
                prompt_parts.append(f"- Your target price: ${target_price}")
            elif self.side == 'seller':
                min_price = self.role_state.get('min_price', 0)
                prompt_parts.append(f"- Your minimum price: ${min_price}")
        
        # Current negotiation round
        prompt_parts.append(f"- Negotiation round: {self.turn_count}")
        
        # Previous strategies (last 3)
        if self.strategy_history:
            recent = self.strategy_history[-3:]
            prompt_parts.append(f"- Previous strategies tried: {[s.get('action_indicator') for s in recent]}")
        
        # Partner's last message
        if hasattr(self, 'observation') and self.observation and 'text' in self.observation:
            prompt_parts.append("")
            prompt_parts.append(f'Partner\'s last message: "{self.observation["text"]}"')
        
        # Instructions
        prompt_parts.append("")
        prompt_parts.append("Generate a JSON strategy with these fields:")
        prompt_parts.append("- action_indicator: [PROPOSE/COMMIT/ACK/NACK/NO_DEAL]")
        prompt_parts.append("- target_price: numerical price goal")
        prompt_parts.append("- tone: aggressive/reasonable/sympathetic/firm")
        prompt_parts.append("- tactics: list of tactics to employ")
        prompt_parts.append("- reasoning: why this approach (1 sentence)")
        prompt_parts.append("")
        prompt_parts.append("Respond with ONLY valid JSON.")
        
        return "\n".join(prompt_parts)
    
    def _build_response_prompt(self, strategy: Dict[str, Any]) -> str:
        """Build the prompt for response generation using dialogue flow manager."""
        # Extract action from strategy
        action = self.dialogue_flow.parse_strategy_action(strategy)
        if not action or not self.dialogue_flow.validate_action(action):
            logging.warning('Fallback triggered - using PROPOSE tag instead of strategy generated tag.')
            action = 'PROPOSE'
            strategy['action_indicator'] = action
        
        # Build context for dialogue flow manager
        context = {
            'side': self.side if self.side else "negotiator",
            'state': self.current_state,
            'history': self.negotiation_history[-5:] if self.negotiation_history else [],
            'dialogue_id': getattr(self, 'dialogue_id', 'unknown'),
            'turn': self.turn_count
        }
        
        # Check if we need partner's terms for ACK
        if action == 'ACK' and hasattr(self, 'terms_buffer'):
            context['partner_terms'] = self.terms_buffer
        
        # Use dialogue flow manager's response prompt generation
        return self.dialogue_flow.generate_response_prompt(strategy, action, context)
    
    def _parse_model_output(self, output: str) -> Tuple[Optional[NegotiationAction], str]:
        """Parse model output to extract action and clean text."""
        # First try to parse any existing action
        action, text = self.parse_negotiation_action(output)
        
        # If no action found and we're in a negotiation, default to PROPOSE
        if action is None and self.negotiation_state.value != 'initializing':
            action = NegotiationAction.PROPOSE
            text = output
        
        return action, text
    
    def _generate_response(self, prompt: str, temperature: Optional[float] = None, 
                          max_tokens: Optional[int] = None) -> Tuple[str, int, int]:
        """Generate response using Ollama API. Returns (response, prompt_tokens, output_tokens)."""
        try:
            # Build the request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature if temperature is not None else self.temperature,
                    "top_p": self.top_p,
                    "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
                }
            }
            
            # Add system prompt if provided
            if self.system_prompt:
                payload["system"] = self.system_prompt
            
            # Make request to Ollama
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('response', '')
                # Estimate tokens (simple approximation)
                prompt_tokens = len(prompt.split()) * 1.3
                output_tokens = len(text.split()) * 1.3
                return text, int(prompt_tokens), int(output_tokens)
            else:
                logger.error(f"Ollama API error: HTTP {response.status_code}")
                fallback = "[PROPOSE] I'm having technical difficulties. Let's continue negotiating."
                return fallback, 0, len(fallback.split())
                
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {self.timeout} seconds")
            fallback = "[PROPOSE] Let me think about that offer."
            return fallback, 0, len(fallback.split())
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            fallback = "[PROPOSE] Let's continue with the negotiation."
            return fallback, 0, len(fallback.split())
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            fallback = "[PROPOSE] I need to reconsider."
            return fallback, 0, len(fallback.split())
    
    def generate_strategy(self) -> Dict[str, Any]:
        """Generate a negotiation strategy using internal prompt generation."""
        # Always use internal prompt generation
        prompt = self._build_strategy_prompt()
            
        start_time = time.time()
        
        # Generate with lower temperature for more consistent strategies
        response_text, prompt_tokens, output_tokens = self._generate_response(
            prompt, 
            temperature=self.strategy_temperature,
            max_tokens=self.strategy_max_tokens
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Track metrics
        self.token_metrics.track_strategy_generation(
            prompt_tokens, output_tokens, duration_ms
        )
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            json_match = response_text
            if '{' in response_text and '}' in response_text:
                start = response_text.index('{')
                end = response_text.rindex('}') + 1
                json_match = response_text[start:end]
            
            strategy = json.loads(json_match)
            
            # Validate strategy
            is_valid, errors = self.validate_strategy(strategy)
            if not is_valid:
                logger.warning(f"Invalid strategy generated: {errors}")
                # Return default strategy
                strategy = self._get_default_strategy()
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse strategy JSON: {e}")
            # Fallback to default strategy
            strategy = self._get_default_strategy()
        
        # Add to strategy history
        self.strategy_history.append(strategy)
        if len(self.strategy_history) > 10:  # Keep last 10 strategies
            self.strategy_history.pop(0)
        
        return strategy
    
    def generate_response(self, strategy: Dict[str, Any]) -> str:
        """Generate a response based on strategy using internal prompt generation.
        
        Args:
            strategy: Strategy dict containing action_indicator and other details
            
        Returns:
            Generated response text
        """
        # Generate prompt internally based on strategy
        prompt = self._build_response_prompt(strategy)
        
        start_time = time.time()
        
        # Generate response
        response_text, prompt_tokens, output_tokens = self._generate_response(
            prompt,
            temperature=self.temperature,
            max_tokens=self.response_max_tokens
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Track metrics
        self.token_metrics.track_response_generation(
            prompt_tokens, output_tokens, duration_ms
        )
        
        # Extract action indicator from strategy for backend use only
        # CRITICAL: Do NOT add tags to the actual response text!
        # Tags are backend-only metadata, never part of agent messages
        action_indicator = 'PROPOSE'  # default
        if strategy and isinstance(strategy, dict):
            action_indicator = strategy.get('action_indicator', 'PROPOSE')
        
        # REMOVED: Tag injection into response text
        # Tags should NEVER appear in agent messages - they're backend only!
        
        return response_text
    
    def _get_default_strategy(self) -> Dict[str, Any]:
        """Get a default strategy when generation fails."""
        # Default based on side and state
        if self.side == 'buyer':
            target = self.role_state.get('target_price', 100)
        else:
            target = self.role_state.get('min_price', 150)
        
        return {
            'action_indicator': 'PROPOSE',
            'target_price': target,
            'tone': 'reasonable',
            'tactics': ['negotiate'],
            'reasoning': 'Continue negotiation toward target price'
        }
    
    def act(self):
        """Generate a response using Ollama (with optional two-part strategy)."""
        if self.use_two_part_response:
            # Phase 1: Generate strategy
            strategy = self.generate_strategy()
            self.current_strategy = strategy
            
            # Phase 2: Generate response based on strategy
            response = self.generate_response(strategy)
            
            # Parse action from strategy (more reliable than parsing from text)
            action_str = strategy.get('action_indicator', 'PROPOSE')
            action = None
            for act in NegotiationAction:
                if act.value == f"[{action_str}]":
                    action = act
                    break
        else:
            # Legacy single-step generation - generate strategy then response
            strategy = self.generate_strategy()
            self.current_strategy = strategy
            response = self.generate_response(strategy)
            
            # Parse action from strategy
            action_str = strategy.get('action_indicator', 'PROPOSE')
            action = None
            for act in NegotiationAction:
                if act.value == f"[{action_str}]":
                    action = act
                    break
        
        # Update our state
        if action:
            self.update_negotiation_state(action, is_self=True)
            self.metrics['action_history'].append({
                'turn': self.turn_count,
                'side': 'self',
                'action': action.value
            })
        
        # Track metrics
        self.track_negotiation_metrics(response, is_outgoing=True)
        
        # Add to history
        self.negotiation_history.append({
            'turn': self.turn_count,
            'is_self': True,
            'text': response,
            'action': action.value if action else None,
            'strategy': self.current_strategy if self.use_two_part_response else None
        })
        
        # Return ParlAI message format
        return {
            'id': self.id,
            'text': response,
            'episode_done': self.negotiation_state.value in ['concluded', 'failed'],
            # Include strategy info for logging
            'strategy': self.current_strategy if self.use_two_part_response else None,
            'token_metrics': self.token_metrics.get_phase_breakdown() if self.use_two_part_response else None,
            # CRITICAL: Store action tag as backend metadata, NOT in text!
            'backend_tag': action_str if 'action_str' in locals() else 'PROPOSE',
            'action_tag': action_str if 'action_str' in locals() else 'PROPOSE'
        }
    
    def batch_act(self, observations):
        """Handle batch inference for multiple conversations."""
        # Process sequentially - Ollama doesn't support true batching
        responses = []
        for obs in observations:
            self.observe(obs)
            responses.append(self.act())
        return responses
    
    def share(self):
        """Share model configuration across agents."""
        shared = super().share()
        # Ollama runs as a service, so no model to share in memory
        # Just share configuration
        shared['ollama_config'] = {
            'model_name': self.model_name,
            'ollama_host': self.ollama_host,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_tokens': self.max_tokens
        }
        return shared