"""
Export built-in ParlAI tasks (Deal-or-No-Deal, Casino) to training format.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
from parlai.core.agents import create_agent

# Set up logging (will use parent logger from data_converter if available)
logger = logging.getLogger(__name__)


class BuiltinTaskExporter:
    """Export ParlAI built-in negotiation tasks."""
    
    def __init__(self, output_dir: str = "data/training/builtin_tasks"):
        """Initialize the exporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Task-specific parsers
        self.task_parsers = {
            'dealnodeal': self._parse_dealnodeal,
            'casino': self._parse_casino
        }
    
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
                        logger.warning(f"Failed to parse line in {filepath}: {e}")
        return data
    
    def export_task(self,
                   task_name: str,
                   max_episodes: int = 1000,
                   datatype: str = 'train') -> List[Dict[str, Any]]:
        """
        Export a single ParlAI task to episode format.
        
        Args:
            task_name: Name of ParlAI task ('dealnodeal', 'casino')
            max_episodes: Maximum episodes to extract
            datatype: Data split ('train', 'valid', 'test')
        
        Returns:
            List of episodes in standard format
        """
        logger.info(f"Exporting {task_name} ({datatype})...")
        
        try:
            # Setup ParlAI task
            parser = ParlaiParser(add_model_args=True, add_parlai_args=True)
            parser.set_params(
                task=task_name,
                datatype=datatype,
                datapath='data/parlai'
            )
            opt = parser.parse_args([], print_args=False)
            
            # Create task world
            agent = create_agent(opt, requireModelExists=False)
            world = create_task(opt, agent)
            
            # Extract episodes
            episodes = []
            episode_count = 0
            
            while episode_count < max_episodes:
                world.parley()
                obs = world.get_acts()[0]
                
                if obs and 'text' in obs:
                    # Convert to episode format
                    episode = self._convert_observation(obs, task_name)
                    if episode:
                        episodes.append(episode)
                        episode_count += 1
                
                if world.epoch_done():
                    logger.info(f"Epoch complete: {episode_count} episodes extracted")
                    break
            
            return episodes
            
        except Exception as e:
            logger.error(f"Error exporting {task_name}: {e}")
            return []
    
    def _convert_observation(self, 
                           obs: Dict[str, Any],
                           task_name: str) -> Optional[Dict[str, Any]]:
        """
        Convert ParlAI observation to episode format.
        
        Args:
            obs: ParlAI observation dict
            task_name: Name of the task
        
        Returns:
            Episode dict or None if conversion fails
        """
        text = obs.get('text', '')
        labels = obs.get('labels', obs.get('eval_labels', []))
        
        if not text:
            return None
        
        # Parse using task-specific parser
        parser = self.task_parsers.get(task_name, self._parse_generic)
        dialogue, state = parser(text, labels)
        
        if not dialogue:
            return None
        
        # Create episode
        episode = {
            'state': state,
            'dialogue': dialogue,
            'meta': {
                'source_task': task_name,
                'episode_id': f"{task_name}_{hash(text)%100000}",
                'datatype': obs.get('datatype', 'train'),
                'original_text': text[:500]  # Store snippet for debugging
            }
        }
        
        return episode
    
    def _parse_dealnodeal(self, 
                         text: str,
                         labels: List[str]) -> Tuple[List[Dict], Dict]:
        """
        Parse Deal-or-No-Deal task format.
        
        Format example:
        context: book=3 book_value=1 hat=2 hat_value=5 ball=1 ball_value=4
        dialogue:
        YOU: I'd like 2 books and 1 hat
        THEM: I need the hat, how about I take 1 hat and you get everything else?
        """
        dialogue = []
        state: Dict[str, Any] = {'task': 'dealnodeal', 'category': 'negotiation'}
        
        # Extract context
        context_match = re.search(r'context:\s*(.+?)(?:\n|$)', text)
        if context_match:
            context_str = context_match.group(1)
            # Parse items and values
            items = {}
            for item_match in re.finditer(r'(\w+)=(\d+)\s+(\w+)_value=(\d+)', context_str):
                item_name = item_match.group(1)
                quantity = int(item_match.group(2))
                value = int(item_match.group(4))
                items[item_name] = {
                    'quantity': quantity,
                    'value': value
                }
            state['items'] = items
        
        # Extract dialogue
        dialogue_match = re.search(r'dialogue:\s*(.+)', text, re.DOTALL)
        if dialogue_match:
            dialogue_text = dialogue_match.group(1)
            
            # Parse turns
            for line in dialogue_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('YOU:'):
                    turn = {
                        'text': line[4:].strip(),
                        'side': 'agent',
                        'action': 'PROPOSE'
                    }
                    dialogue.append(turn)
                elif line.startswith('THEM:'):
                    turn = {
                        'text': line[5:].strip(),
                        'side': 'partner',
                        'action': 'PROPOSE'
                    }
                    dialogue.append(turn)
        
        # Add response if available
        if labels and dialogue:
            # Last turn gets the response
            dialogue[-1]['response'] = labels[0]
        elif labels and not dialogue:
            # Create a single turn with response
            dialogue.append({
                'text': '',
                'response': labels[0],
                'side': 'agent',
                'action': 'PROPOSE'
            })
        
        return dialogue, state
    
    def _parse_casino(self,
                     text: str,
                     labels: List[str]) -> Tuple[List[Dict], Dict]:
        """
        Parse Casino task format.
        
        Similar to Deal-or-No-Deal but with different context format.
        """
        dialogue = []
        state: Dict[str, Any] = {'task': 'casino', 'category': 'negotiation'}
        
        # Split into lines
        lines = text.split('\n')
        current_context = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for context markers
            if 'context' in line.lower() or 'setting' in line.lower():
                # Extract any key-value pairs
                for match in re.finditer(r'(\w+)[:\s]*=\s*([^\s,]+)', line):
                    key = match.group(1)
                    value = match.group(2)
                    current_context[key] = value
            
            # Check for dialogue section
            elif 'dialogue' in line.lower() or ':' in line:
                # Parse speaker and text
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        speaker = parts[0].strip().upper()
                        content = parts[1].strip()
                        
                        if speaker in ['YOU', 'AGENT', 'PLAYER1']:
                            side = 'agent'
                        elif speaker in ['THEM', 'PARTNER', 'PLAYER2']:
                            side = 'partner'
                        else:
                            side = 'unknown'
                        
                        turn = {
                            'text': content,
                            'side': side,
                            'action': 'PROPOSE'
                        }
                        dialogue.append(turn)
        
        # Store context in state
        if current_context:
            state['context'] = current_context
        
        # Add response
        if labels and dialogue:
            dialogue[-1]['response'] = labels[0]
        elif labels:
            dialogue.append({
                'text': text.split('\n')[-1] if '\n' in text else '',
                'response': labels[0],
                'side': 'agent',
                'action': 'PROPOSE'
            })
        
        return dialogue, state
    
    def _parse_generic(self,
                      text: str,
                      labels: List[str]) -> Tuple[List[Dict], Dict]:
        """Generic parser for unknown task formats."""
        dialogue = []
        state: Dict[str, Any] = {'category': 'negotiation'}
        
        # Simple line-by-line parsing
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                dialogue.append({
                    'text': line,
                    'side': 'unknown',
                    'action': 'PROPOSE'
                })
        
        # Add response
        if labels and dialogue:
            dialogue[-1]['response'] = labels[0]
        
        return dialogue, state
    
    def export_all_tasks(self,
                        task_names: Optional[List[str]] = None,
                        max_episodes_per_task: int = 1000) -> Dict[str, str]:
        """
        Export multiple tasks and save to files.
        
        Args:
            task_names: List of tasks to export (default: ['dealnodeal', 'casino'])
            max_episodes_per_task: Max episodes per task
        
        Returns:
            Dict mapping task names to output file paths
        """
        if task_names is None:
            task_names = ['dealnodeal', 'casino']
        
        output_files = {}
        
        for task_name in task_names:
            episodes = self.export_task(task_name, max_episodes_per_task)
            
            if episodes:
                # Save to file
                output_path = self.output_dir / f"{task_name}_episodes.jsonl"
                self.save_jsonl(episodes, output_path)
                output_files[task_name] = str(output_path)
                
                logger.info(f"Exported {len(episodes)} episodes from {task_name}")
                logger.info(f"Saved to {output_path}")
            else:
                logger.warning(f"No episodes exported from {task_name}")
        
        return output_files