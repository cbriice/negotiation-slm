import logging
import json
import time
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft.peft_model import PeftModel
from torch.amp.autocast_mode import autocast

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

# Suppress urllib3 debug logs
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

# NOTE: OllamaNegotiator actually has HuggingFace logic and model pulling inside for QLoRA compatibility. Name of the class was preserved to avoid import conflicts

class OllamaNegotiator(NegotiatorBase):
    """Ollama model wrapped as a ParlAI negotiation agent."""
    
    @classmethod
    def add_cmdline_args(cls, argparser, partial_opt=None):
        """Add command-line arguments specific to this agent."""
        agent = argparser.add_argument_group('HF Negotiator Arguments')
        agent.add_argument(
            '--model-name',
            type=str,
            default='mistralai/Mistral-7B-v0.1',
            help='HuggingFace model name'
        )
        agent.add_argument(
            '--adapter-path',
            type=str,
            default=None,
            help='Path to LoRA adapter checkpoint'
        )
        agent.add_argument(
            '--use-latest-checkpoint',
            action='store_true',
            default=True,
            help='Automatically load the latest adapter checkpoint'
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
        self.model_name = opt.get('model_name', 'mistralai/Mistral-7B-v0.1')
        self.adapter_path = opt.get('adapter_path', None)
        self.use_latest_checkpoint = opt.get('use_latest_checkpoint', True)
        self.temperature = opt.get('temperature', MODEL_TEMP)
        self.top_p = opt.get('top_p', 0.9)
        self.max_tokens = opt.get('max_tokens', 256)
        self.system_prompt = opt.get('system_prompt', None)

        # Load HF model and tokenizer
        if shared and 'model' in shared:
            self.model = shared['model']
            self.tokenizer = shared['tokenizer']
            logger.info("Using shared model instance")
        else:
            self._load_hf_model()
        
        # Two-part response configuration
        self.use_two_part_response = opt.get('use_two_part_response', USE_TWO_PART_RESPONSE)
        logger.info(f"OllamaNegotiator initialized with use_two_part_response={self.use_two_part_response}")
        self.strategy_temperature = opt.get('strategy_temperature', STRATEGY_TEMPERATURE)
        self.strategy_max_tokens = opt.get('strategy_max_tokens', STRATEGY_MAX_TOKENS)
        self.response_max_tokens = opt.get('response_max_tokens', RESPONSE_MAX_TOKENS)
        
        # Enhanced token tracking
        self.token_metrics = TokenMetrics()
        
        # Dialogue flow manager for prompt generation
        self.dialogue_flow = DialogueFlowManager()
        
        # Set device and optimize CUDA settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            # Enable TF32 for better GPU utilization on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Set cudnn benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
        logger.info(f"Using device: {self.device}")
    
    def _load_hf_model(self):
        """Load HuggingFace model with optional LoRA adapters."""
        logger.info(f"Loading HF model: {self.model_name}")

        # Auto-load latest checkpoint if requested
        if self.use_latest_checkpoint and not self.adapter_path:
            from config.training_config import get_latest_adapter_path
            latest_adapter = get_latest_adapter_path()
            if latest_adapter:
                self.adapter_path = str(latest_adapter)
                logger.info(f"Auto-loading latest adapter: {self.adapter_path}")

        # Set up quantization config for 4-bit loading
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side='left',
            )

            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Calculate max_memory for GPU (use 85% like in training)
            if torch.cuda.is_available():
                # Get total VRAM in bytes
                total_memory = torch.cuda.get_device_properties(0).total_memory
                # Use 85% for model, leave 15% for activations
                usable_memory_gb = (total_memory * 0.85) / (1024**3)
                max_memory = {0: f"{usable_memory_gb:.2f}GB"}
                logger.info(f"Setting max GPU memory to {usable_memory_gb:.2f}GB (85% of total)")
            else:
                max_memory = None

            # Load base model with quantization and Flash Attention 2
            logger.info("Loading base model with Flash Attention 2...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",  # Enable Flash Attention 2
            )

            # Verify Flash Attention 2 is active on base model
            if hasattr(self.model, 'config') and hasattr(self.model.config, '_attn_implementation'):
                logger.info(f"✓ Base model loaded with attention: '{self.model.config._attn_implementation}'")
            else:
                logger.info("Base model loaded (FA2 status cannot be verified from config)")

            # Load LoRA adapter if provided
            if self.adapter_path:
                adapter_path = Path(self.adapter_path)
                if adapter_path.exists():
                    logger.info(f"Loading LoRA adapter from: {adapter_path}")
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        str(adapter_path),
                        torch_dtype=torch.bfloat16,
                    )
                    logger.info("LoRA adapter loaded successfully")

                    # Verify Flash Attention persists through PEFT
                    logger.info("Verifying Flash Attention 2 configuration after PEFT loading...")

                    # Try to find the config in the PEFT model structure
                    config = None
                    if hasattr(self.model, 'config'):
                        config = self.model.config
                    elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'config'):
                        config = self.model.base_model.config

                    if config and hasattr(config, '_attn_implementation'):
                        logger.info(f"✓ FA2 Status: Attention implementation = '{config._attn_implementation}'")
                        if config._attn_implementation == "flash_attention_2":
                            logger.info("✓ Flash Attention 2 is ACTIVE after PEFT loading")
                        else:
                            logger.warning(f"⚠ Flash Attention 2 NOT active. Current: {config._attn_implementation}")
                    else:
                        logger.warning("⚠ Could not verify attention implementation after PEFT loading")
                else:
                    logger.warning(f"Adapter path does not exist: {adapter_path}")

            # Set model to eval mode for faster inference
            self.model.eval()
            logger.info("Model loaded successfully in eval mode")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
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
        """Generate response using HuggingFace model. Returns (response, prompt_tokens, output_tokens)."""
        try:
            # Prepare full prompt with system prompt if provided
            full_prompt = prompt
            if self.system_prompt:
                full_prompt = f"{self.system_prompt}\n\n{prompt}"

            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)

            # Count prompt tokens
            prompt_tokens = inputs['input_ids'].shape[1]

            # Generate with mixed precision for better GPU utilization
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                    temperature=temperature if temperature is not None else self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache for fast autoregressive decoding
                )

            # Decode the response (only the new tokens)
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Count output tokens
            output_tokens = outputs.shape[1] - prompt_tokens

            return response.strip(), prompt_tokens, output_tokens

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory. Consider reducing batch size or max tokens.")
            torch.cuda.empty_cache()
            fallback = "Let me reconsider that offer."
            return fallback, 0, len(self.tokenizer.encode(fallback))
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            fallback = "Let's continue with the negotiation."
            return fallback, 0, len(self.tokenizer.encode(fallback))

    def _batch_generate(self, prompts: List[str], temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> List[str]:
        """
        Generate responses for multiple prompts in parallel batch.
        Maximizes GPU utilization by processing all prompts together.
        """
        if not prompts:
            return []

        try:
            # Add system prompt if provided
            if self.system_prompt:
                prompts = [f"{self.system_prompt}\n\n{p}" for p in prompts]

            # Batch tokenize all prompts at once
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,  # Pad to same length for batching
            ).to(self.device)

            # Track input lengths for proper decoding
            input_lengths = inputs['attention_mask'].sum(dim=1)

            # Batch generate with mixed precision
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                    temperature=temperature if temperature is not None else self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

            # Decode all outputs in parallel
            responses = []
            for i in range(len(prompts)):
                # Extract only generated tokens
                generated_tokens = outputs[i][input_lengths[i]:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                responses.append(response.strip())

            return responses

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU OOM in batch generation. Falling back to sequential.")
            torch.cuda.empty_cache()
            # Fallback to sequential processing
            responses = []
            for prompt in prompts:
                response, _, _ = self._generate_response(prompt, temperature, max_tokens)
                responses.append(response)
            return responses
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            # Return fallback responses
            return ["Let's continue negotiating." for _ in prompts]

    def _parse_strategy_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON strategy from text."""
        try:
            # Extract JSON from response
            if '{' in text and '}' in text:
                start = text.index('{')
                end = text.rindex('}') + 1
                json_str = text[start:end]
                return json.loads(json_str)
            return json.loads(text)
        except:
            return self._get_default_strategy()
    
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
            # Direct generation without strategy
            # Pass None strategy to dialogue flow manager
            self.current_strategy = None

            # Build context for dialogue flow manager (without strategy)
            context = {
                'side': self.side if self.side else "negotiator",
                'state': self.current_state,
                'history': self.negotiation_history[-5:] if self.negotiation_history else [],
                'dialogue_id': getattr(self, 'dialogue_id', 'unknown'),
                'turn': self.turn_count,
                'role_state': self.role_state  # Pass role state for target price
            }

            # Use dialogue flow manager with None strategy
            prompt = self.dialogue_flow.generate_response_prompt(None, 'PROPOSE', context)

            start_time = time.time()
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

            response = response_text

            # For direct generation (no strategy), default to PROPOSE action
            # Actions are backend metadata, not parsed from message text
            action = NegotiationAction.PROPOSE
            action_str = action.value.strip('[]')
        
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
        """Handle TRUE batch inference for multiple conversations."""
        if not observations:
            return []

        batch_size = len(observations)

        # Process all observations first
        for obs in observations:
            self.observe(obs)

        if self.use_two_part_response:
            # Phase 1: Batch generate strategies for ALL conversations
            strategy_prompts = []
            for _ in range(batch_size):
                strategy_prompts.append(self._build_strategy_prompt())

            # Batch tokenize and generate strategies in parallel
            strategies_texts = self._batch_generate(
                strategy_prompts,
                temperature=self.strategy_temperature,
                max_tokens=self.strategy_max_tokens
            )

            # Parse strategies
            strategies = []
            for text in strategies_texts:
                try:
                    strategy = self._parse_strategy_json(text)
                    strategies.append(strategy)
                except:
                    strategies.append(self._get_default_strategy())

            # Phase 2: Batch generate responses based on strategies
            response_prompts = []
            for strategy in strategies:
                response_prompts.append(self._build_response_prompt(strategy))

            responses_texts = self._batch_generate(
                response_prompts,
                temperature=self.temperature,
                max_tokens=self.response_max_tokens
            )

            # Build final responses
            responses = []
            for i, (strategy, response_text) in enumerate(zip(strategies, responses_texts)):
                action_str = strategy.get('action_indicator', 'PROPOSE')
                responses.append({
                    'id': self.id,
                    'text': response_text,
                    'episode_done': False,
                    'strategy': strategy,
                    'backend_tag': action_str,
                    'action_tag': action_str
                })
        else:
            # Direct batch generation without strategies
            prompts = []
            for _ in range(batch_size):
                context = {
                    'side': self.side if self.side else "negotiator",
                    'state': self.current_state,
                    'history': self.negotiation_history[-5:] if self.negotiation_history else [],
                    'dialogue_id': getattr(self, 'dialogue_id', 'unknown'),
                    'turn': self.turn_count,
                    'role_state': self.role_state
                }
                prompt = self.dialogue_flow.generate_response_prompt(None, 'PROPOSE', context)
                prompts.append(prompt)

            # Batch generate all responses
            responses_texts = self._batch_generate(
                prompts,
                temperature=self.temperature,
                max_tokens=self.response_max_tokens
            )

            # Build responses
            responses = []
            for response_text in responses_texts:
                responses.append({
                    'id': self.id,
                    'text': response_text,
                    'episode_done': False,
                    'backend_tag': 'PROPOSE',
                    'action_tag': 'PROPOSE'
                })

        return responses
    
    def share(self):
        """Share model configuration across agents."""
        shared = super().share()
        # Share the loaded model and tokenizer for efficiency
        shared['model'] = self.model
        shared['tokenizer'] = self.tokenizer
        shared['hf_config'] = {
            'model_name': self.model_name,
            'adapter_path': self.adapter_path,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_tokens': self.max_tokens
        }
        return shared