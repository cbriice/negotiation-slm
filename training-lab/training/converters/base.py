"""
Base converter functionality and utilities.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Set up logging
logger = logging.getLogger(__name__)


class BaseConverter:
    """Base class for data format converters."""
    
    def __init__(self, output_dir: str = "data/training"):
        """
        Initialize the base converter.
        
        Args:
            output_dir: Directory to save converted training data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_jsonl(self, data: List[Dict], filepath: Path) -> None:
        """Save data to JSONL file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    def load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load data from JSONL file."""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")
        return data
    
    def extract_dialogue_history(self, dialogue: List[Dict], 
                                turn_idx: int, 
                                window_size: int = 5) -> List[Dict]:
        """
        Extract conversation history up to a given turn.
        
        Args:
            dialogue: Full dialogue list
            turn_idx: Current turn index
            window_size: Number of recent turns to include
            
        Returns:
            List of historical dialogue turns
        """
        history = []
        start_idx = max(0, turn_idx - window_size)
        
        for hist_idx in range(start_idx, turn_idx):
            if hist_idx < len(dialogue):
                hist_turn = dialogue[hist_idx]
                history.append({
                    'side': hist_turn.get('side', 'unknown'),
                    'text': hist_turn.get('text', ''),
                    'action': hist_turn.get('action', 'PROPOSE')
                })
        
        return history