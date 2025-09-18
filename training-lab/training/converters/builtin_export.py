"""
Export built-in ParlAI tasks (Deal-or-No-Deal, Casino) to training format.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re
from parlai.core.params import ParlaiParser
from parlai.core.teachers import create_task_agent_from_taskname

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
            'dealnodeal': self._parse_dealnodeal
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
            # Setup ParlAI parser options
            parser = ParlaiParser(add_parlai_args=True, add_model_args=False)
            opt = parser.parse_args([], print_args=False)
            opt['task'] = task_name
            opt['datatype'] = datatype
            opt['datapath'] = 'data/parlai'
            opt['interactive_task'] = False

            # Create teacher agent directly from task
            teacher = create_task_agent_from_taskname(opt)[0]

            # Extract full episodes
            episodes = []
            episode_count = 0
            current_episode_data = []
            current_state = {}

            while not teacher.epoch_done() and episode_count < max_episodes:
                # Get example from teacher
                example = teacher.act()

                if example:
                    # Check if this starts a new episode (has 'items' in dealornodeal)
                    if 'items' in example:
                        # Process previous episode if exists
                        if current_episode_data:
                            episode = self._build_full_episode(current_episode_data, current_state, task_name)
                            if episode:
                                episodes.append(episode)
                                episode_count += 1
                                if episode_count % 100 == 0:
                                    logger.info(f"  Extracted {episode_count} episodes...")

                        # Start new episode
                        current_state = {'items': example.get('items', {})}
                        current_episode_data = [example]
                    else:
                        # Add to current episode
                        current_episode_data.append(example)

                    # Check if episode is done
                    if example.get('episode_done', False):
                        # Process completed episode
                        if current_episode_data:
                            episode = self._build_full_episode(current_episode_data, current_state, task_name)
                            if episode:
                                episodes.append(episode)
                                episode_count += 1
                                if episode_count % 100 == 0:
                                    logger.info(f"  Extracted {episode_count} episodes...")
                        # Reset for next episode
                        current_episode_data = []
                        current_state = {}

                # Prepare for next example
                teacher.observe({'episode_done': example.get('episode_done', False)})

            # Process any remaining episode
            if current_episode_data:
                episode = self._build_full_episode(current_episode_data, current_state, task_name)
                if episode:
                    episodes.append(episode)
                    episode_count += 1

            logger.info(f"Extracted total of {episode_count} episodes from {task_name}")
            return episodes

        except Exception as e:
            logger.error(f"Error exporting {task_name}: {e}")
            import traceback
            traceback.print_exc()
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
        # Get text - handle different field names from teacher
        text = obs.get('text', '')

        # Get labels/response
        labels = obs.get('labels', obs.get('eval_labels', []))
        if not labels:
            # Try to get labels_vec if text labels not available
            labels = obs.get('labels_vec', [])

        if not text and not labels:
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
                'episode_id': f"{task_name}_{abs(hash(text))%100000}",
                'datatype': obs.get('datatype', 'train'),
                'episode_done': obs.get('episode_done', False)
            }
        }

        # Add original text snippet for debugging if present
        if text:
            episode['meta']['original_text'] = text[:500]

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
    
    def _build_full_episode(self,
                           episode_data: List[Dict[str, Any]],
                           state: Dict[str, Any],
                           task_name: str) -> Optional[Dict[str, Any]]:
        """
        Build a complete episode from collected turns.

        Args:
            episode_data: List of turns from teacher
            state: Episode state (e.g., items)
            task_name: Name of the task

        Returns:
            Complete episode dict or None
        """
        if not episode_data:
            return None

        dialogue = []
        full_text = []

        for i, turn in enumerate(episode_data):
            text = turn.get('text', '')
            labels = turn.get('labels', [])

            if text:
                full_text.append(text)

            # Build dialogue turn
            if i == 0 and 'items' in turn:
                # First turn with context - use label as first agent response
                if labels:
                    dialogue.append({
                        'text': text,
                        'response': labels[0],
                        'side': 'agent',
                        'action': 'PROPOSE'
                    })
            else:
                # Regular dialogue turn
                if labels:
                    dialogue.append({
                        'text': text,
                        'response': labels[0],
                        'side': 'agent' if i % 2 == 0 else 'partner',
                        'action': 'PROPOSE'
                    })

        if not dialogue:
            return None

        # Create full episode
        episode_state = {'task': task_name, 'category': 'negotiation'}
        episode_state.update(state)

        episode = {
            'state': episode_state,
            'dialogue': dialogue,
            'meta': {
                'source_task': task_name,
                'episode_id': f"{task_name}_{abs(hash(''.join(full_text)))%100000}",
                'datatype': episode_data[0].get('datatype', 'train'),
                'num_turns': len(dialogue)
            }
        }

        return episode

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