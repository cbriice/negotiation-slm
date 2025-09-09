import logging
import json
from typing import Optional, Tuple
from pathlib import Path
import requests
import time

from agents.negotiator_base import NegotiatorBase, NegotiationAction

# Set up logging for agent-related logs
Path('logs').mkdir(exist_ok=True)

# Configure logging with 'w' mode to clear log on each run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/agent_logs.log', mode='w'),
        logging.StreamHandler()
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
            default=0.7,
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
        self.temperature = opt.get('temperature', 0.5)
        self.top_p = opt.get('top_p', 0.9)
        self.max_tokens = opt.get('max_tokens', 256)
        self.timeout = opt.get('timeout', 30)
        self.system_prompt = opt.get('system_prompt', None)
        
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
    
    def _build_prompt(self) -> str:
        """Build the prompt for the model based on negotiation context."""
        prompt_parts = []
        
        # Add role and state information
        if self.current_state:
            side = self.side if self.side else "negotiator"
            prompt_parts.append(f"You are a {side} in a negotiation.")
            
            # Add item information
            if 'item' in self.current_state:
                item = self.current_state['item']
                prompt_parts.append(f"Item: {item.get('name', 'Unknown')}")
                prompt_parts.append(f"Condition: {item.get('condition', 'Unknown')}")
                prompt_parts.append(f"List Price: ${item.get('list_price', 0)}")
            
            # Add role-specific constraints
            if self.role_state:
                if self.side == 'buyer':
                    max_price = self.role_state.get('max_price', 0)
                    target_price = self.role_state.get('target_price', 0)
                    prompt_parts.append(f"Your maximum price: ${max_price}")
                    prompt_parts.append(f"Your target price: ${target_price}")
                elif self.side == 'seller':
                    min_price = self.role_state.get('min_price', 0)
                    prompt_parts.append(f"Your minimum price: ${min_price}")
        
        # Add conversation history
        if self.negotiation_history:
            prompt_parts.append("\nConversation history:")
            # Take last 5 turns for context
            recent_history = self.negotiation_history[-5:]
            for turn in recent_history:
                role = "You" if turn.get('is_self', False) else "Partner"
                prompt_parts.append(f"{role}: {turn.get('text', '')}")
        
        # Add current observation
        if hasattr(self, 'observation') and self.observation:
            if 'text' in self.observation and self.observation['text']:
                prompt_parts.append(f"\nPartner: {self.observation['text']}")
        
        # Add instruction for response
        prompt_parts.append("\nYour response (use [PROPOSE], [COMMIT], [ACK], [NACK], or [NO_DEAL]):")
        
        return "\n".join(prompt_parts)
    
    def _parse_model_output(self, output: str) -> Tuple[Optional[NegotiationAction], str]:
        """Parse model output to extract action and clean text."""
        # First try to parse any existing action
        action, text = self.parse_negotiation_action(output)
        
        # If no action found and we're in a negotiation, default to PROPOSE
        if action is None and self.negotiation_state.value != 'initializing':
            action = NegotiationAction.PROPOSE
            text = output
        
        return action, text
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using Ollama API."""
        try:
            # Build the request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
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
                return result.get('response', '')
            else:
                logger.error(f"Ollama API error: HTTP {response.status_code}")
                return "[PROPOSE] I'm having technical difficulties. Let's continue negotiating."
                
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {self.timeout} seconds")
            return "[PROPOSE] Let me think about that offer."
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            return "[PROPOSE] Let's continue with the negotiation."
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return "[PROPOSE] I need to reconsider."
    
    def act(self):
        """Generate a response using Ollama."""
        # Build prompt
        prompt = self._build_prompt()
        
        # Generate response
        response = self._generate_response(prompt)
        
        # Parse action and text
        action, text = self._parse_model_output(response)
        
        # Format response with proper action tag if needed
        if action and action.value not in response:
            response = self.format_response(text, action)
        
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
            'action': action.value if action else None
        })
        
        # Return ParlAI message format
        return {
            'id': self.id,
            'text': response,
            'episode_done': self.negotiation_state.value in ['concluded', 'failed']
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