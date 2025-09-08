import torch
from typing import Optional, Tuple
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)

from agents.negotiator_base import NegotiatorBase, NegotiationAction

class HFNegotiator(NegotiatorBase):
    """HuggingFace model wrapped as a ParlAI negotiation agent."""
    
    @classmethod
    def add_cmdline_args(cls, argparser, partial_opt=None):
        """Add command-line arguments specific to this agent."""
        agent = argparser.add_argument_group('HF Negotiator Arguments')
        agent.add_argument(
            '--model-name',
            type=str,
            default='mistralai/Mistral-7B-v0.1',
            help='HuggingFace model name or path to checkpoint'
        )
        agent.add_argument(
            '--lora-checkpoint',
            type=str,
            default=None,
            help='Path to LoRA checkpoint if using fine-tuned model'
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
            '--max-new-tokens',
            type=int,
            default=256,
            help='Maximum new tokens to generate'
        )
        agent.add_argument(
            '--use-4bit',
            type=bool,
            default=True,
            help='Use 4-bit quantization'
        )
        return argparser
    
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        
        # Model configuration
        self.model_name = opt.get('model_name', 'mistralai/Mistral-7B-v0.1')
        self.lora_checkpoint = opt.get('lora_checkpoint', None)
        self.temperature = opt.get('temperature', 0.7)
        self.top_p = opt.get('top_p', 0.9)
        self.max_new_tokens = opt.get('max_new_tokens', 256)
        self.use_4bit = opt.get('use_4bit', True)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if shared:
            # Share model across agents for efficiency
            self.model = shared['model']
            self.tokenizer = shared['tokenizer']
        else:
            # Load model and tokenizer
            self._load_model()
    
    def _load_model(self):
        """Load the HuggingFace model with optional LoRA and quantization."""
        print(f"Loading model: {self.model_name}")
        
        # Quantization config for 4-bit
        bnb_config = None
        if self.use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side='left',
            truncation_side='left'
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        
        # Load LoRA weights if provided
        if self.lora_checkpoint and Path(self.lora_checkpoint).exists():
            print(f"Loading LoRA checkpoint: {self.lora_checkpoint}")
            try:
                from peft.peft_model import PeftModel
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_checkpoint,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                )
            except ImportError:
                print("PEFT not installed, skipping LoRA loading")
        
        # Set to eval mode
        self.model.eval()
        
        # Generation config
        self.generation_config = GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
    
    def share(self):
        """Share model across agents to save memory."""
        shared = super().share()
        shared['model'] = self.model
        shared['tokenizer'] = self.tokenizer
        return shared
    
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
    
    def act(self):
        """Generate a response using the HuggingFace model."""
        # Build prompt
        prompt = self._build_prompt()
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536 - self.max_new_tokens,  # Leave room for generation
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
            )
        
        # Decode response (only the new tokens)
        input_length = inputs['input_ids'].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        
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
        # For now, process sequentially
        # TODO: Implement true batched inference for efficiency
        responses = []
        for obs in observations:
            self.observe(obs)
            responses.append(self.act())
        return responses