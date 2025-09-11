"""Token metrics tracking for two-part response system."""
import time
from typing import Dict, Any, Optional


class TokenMetrics:
    """Track token usage separately for strategy and response phases."""
    
    def __init__(self):
        # Strategy phase
        self.strategy_prompt_tokens = 0
        self.strategy_response_tokens = 0
        self.strategy_generation_time_ms = 0
        
        # Response phase
        self.response_prompt_tokens = 0
        self.response_output_tokens = 0
        self.response_generation_time_ms = 0
        
        # Legacy compatibility
        self.total_tokens_sent = 0
        self.total_tokens_received = 0
    
    @property
    def total_input_tokens(self):
        """Total input tokens across both phases."""
        return self.strategy_prompt_tokens + self.response_prompt_tokens
    
    @property
    def total_output_tokens(self):
        """Total output tokens across both phases."""
        return self.strategy_response_tokens + self.response_output_tokens
    
    @property
    def total_tokens(self):
        """Total tokens across all phases."""
        return self.total_input_tokens + self.total_output_tokens
    
    def track_strategy_generation(self, prompt_tokens: int, output_tokens: int, 
                                 duration_ms: Optional[float] = None):
        """Track metrics for strategy generation phase."""
        self.strategy_prompt_tokens += prompt_tokens
        self.strategy_response_tokens += output_tokens
        if duration_ms:
            self.strategy_generation_time_ms += duration_ms
    
    def track_response_generation(self, prompt_tokens: int, output_tokens: int,
                                 duration_ms: Optional[float] = None):
        """Track metrics for response generation phase."""
        self.response_prompt_tokens += prompt_tokens
        self.response_output_tokens += output_tokens
        if duration_ms:
            self.response_generation_time_ms += duration_ms
        
        # Update legacy metrics
        self.total_tokens_sent += output_tokens
        self.total_tokens_received += prompt_tokens
    
    def get_phase_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown by phase."""
        return {
            'strategy': {
                'prompt_tokens': self.strategy_prompt_tokens,
                'output_tokens': self.strategy_response_tokens,
                'total_tokens': self.strategy_prompt_tokens + self.strategy_response_tokens,
                'generation_time_ms': self.strategy_generation_time_ms
            },
            'response': {
                'prompt_tokens': self.response_prompt_tokens,
                'output_tokens': self.response_output_tokens,
                'total_tokens': self.response_prompt_tokens + self.response_output_tokens,
                'generation_time_ms': self.response_generation_time_ms
            },
            'total': {
                'input_tokens': self.total_input_tokens,
                'output_tokens': self.total_output_tokens,
                'all_tokens': self.total_tokens,
                'generation_time_ms': self.strategy_generation_time_ms + self.response_generation_time_ms
            }
        }
    
    def reset(self):
        """Reset all metrics."""
        self.strategy_prompt_tokens = 0
        self.strategy_response_tokens = 0
        self.strategy_generation_time_ms = 0
        self.response_prompt_tokens = 0
        self.response_output_tokens = 0
        self.response_generation_time_ms = 0
        self.total_tokens_sent = 0
        self.total_tokens_received = 0