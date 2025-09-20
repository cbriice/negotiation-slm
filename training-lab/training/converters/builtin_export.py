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

        Collects full multi-turn conversations from Deal or No Deal,
        building conversation history for each turn.

        Args:
            task_name: Name of ParlAI task ('dealnodeal', 'casino')
            max_episodes: Maximum episodes to extract
            datatype: Data split ('train', 'valid', 'test')

        Returns:
            List of episodes in standard format with full dialogue history
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

            # Extract full episodes with conversation history
            episodes = []
            episode_count = 0

            while not teacher.epoch_done() and episode_count < max_episodes:
                # Collect a full episode (multiple turns)
                episode_turns = []
                episode_state = None

                while True:
                    example = teacher.act()
                    if not example:
                        break

                    text = example.get('text', '')
                    labels = example.get('labels', [])
                    episode_done = example.get('episode_done', False)

                    # Store turn data
                    episode_turns.append({
                        'text': text,
                        'labels': labels,
                        'episode_done': episode_done
                    })

                    # Extract state from first turn if it contains item info
                    if not episode_state and 'There are' in text and 'each with a value' in text:
                        episode_state = self._extract_state_from_text(text)

                    # Observe the response to continue conversation
                    if not episode_done and labels:
                        teacher.observe({'text': labels[0]})
                    else:
                        teacher.observe({'episode_done': True})
                        break

                # Build episode with full conversation history
                if episode_turns:
                    episode = self._build_episode_with_history(episode_turns, episode_state, task_name)
                    if episode and len(episode.get('dialogue', [])) > 0:
                        episodes.append(episode)
                        episode_count += 1
                        if episode_count % 100 == 0:
                            logger.info(f"  Extracted {episode_count} episodes...")

            logger.info(f"Extracted total of {episode_count} episodes from {task_name}")
            return episodes

        except Exception as e:
            logger.error(f"Error exporting {task_name}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _extract_state_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract state (item counts and values) from initial text.
        """
        state = {}
        items = {}

        # Format: "There are X book(s), each with a value of Y, ..."
        item_pattern = r'(\d+)\s+(\w+)\(s\),\s+each\s+with\s+a\s+value\s+of\s+(\d+)'
        for match in re.finditer(item_pattern, text):
            count = match.group(1)
            item_name = match.group(2)
            value = match.group(3)
            items[f"{item_name}_cnt"] = count
            items[f"{item_name}_val"] = value

        if items:
            state['items'] = items

        return state

    def _build_episode_with_history(self,
                                   episode_turns: List[Dict],
                                   episode_state: Optional[Dict],
                                   task_name: str) -> Optional[Dict[str, Any]]:
        """
        Build a complete episode with conversation history for each turn.
        """
        if not episode_turns:
            return None

        dialogue = []
        conversation_history = []

        # Process each turn, building up conversation history
        for i, turn in enumerate(episode_turns):
            text = turn['text']
            labels = turn['labels']

            # Skip the initial context turn if it's just setup
            if i == 0 and 'Negotiate with your opponent' in text:
                # Extract clean partner message after context
                parts = text.split('\n', 1)
                if len(parts) > 1:
                    partner_msg = parts[1].strip()
                else:
                    partner_msg = ''
            else:
                partner_msg = text

            # Clean and validate messages
            partner_msg = self._sanitize_dialogue_text(partner_msg)

            # Process labels as agent responses
            if labels and labels[0] != '<selection>':
                agent_response = labels[0]

                # Skip item allocation responses (these are backend-only)
                if agent_response.startswith('item0=') or 'item1=' in agent_response or 'item2=' in agent_response:
                    continue

                # Sanitize the response
                agent_response = self._sanitize_dialogue_text(agent_response)

                # Skip empty responses
                if not agent_response:
                    continue

                # Create dialogue turn with accumulated history
                dialogue_turn = {
                    'text': partner_msg,
                    'response': agent_response,
                    'side': 'agent',
                    'action': 'PROPOSE',
                    'conversation_history': list(conversation_history)  # Copy current history
                }
                dialogue.append(dialogue_turn)

                # Update conversation history for next turn
                if partner_msg:
                    conversation_history.append({
                        'side': 'partner',
                        'text': partner_msg
                    })
                conversation_history.append({
                    'side': 'agent',
                    'text': agent_response
                })

        if not dialogue:
            return None

        # Build complete episode
        state = {
            'task': task_name,
            'category': 'negotiation'
        }
        if episode_state:
            state.update(episode_state)

        episode = {
            'state': state,
            'dialogue': dialogue,
            'meta': {
                'source_task': task_name,
                'episode_id': f"{task_name}_{abs(hash(str(episode_turns)))%100000}",
                'datatype': 'train',
                'num_turns': len(dialogue)
            }
        }

        return episode
    
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
    
    def _sanitize_dialogue_text(self, text: str) -> str:
        """
        Sanitize dialogue text by removing markup, metadata, and prompts.

        Args:
            text: Raw text that may contain prompts or metadata

        Returns:
            Clean dialogue text suitable for training
        """
        # Remove any selection markers or special tokens
        text = re.sub(r'<selection>', '', text)
        text = re.sub(r'<eos>', '', text)
        text = re.sub(r'</*\w+>', '', text)  # Remove any XML-like tags

        # Remove prompt-like content
        prompt_patterns = [
            r'You are a.*?\.',  # "You are a buyer/seller..."
            r'Your target.*?\.',  # "Your target price is..."
            r'Make a negotiation.*?\.',  # "Make a negotiation offer..."
            r'##.*?\n',  # Markdown headers
            r'###.*?\n',  # Markdown subheaders
            r'\[.*?\]',  # Bracketed action tags like [PROPOSE]
        ]

        for pattern in prompt_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n+', '\n', text)
        text = text.strip()

        return text

    def _parse_dealnodeal(self,
                         text: str,
                         labels: List[str]) -> Tuple[List[Dict], Dict]:
        """
        Parse Deal-or-No-Deal task format.

        The data contains conversations where the first message includes item info,
        and subsequent messages are the negotiation dialogue.
        """
        dialogue = []
        state: Dict[str, Any] = {'task': 'dealnodeal', 'category': 'negotiation'}

        # First, try to extract items from the initial context message
        # Format: "There are X book(s), each with a value of Y, ..."
        items = {}
        item_pattern = r'(\d+)\s+(\w+)\(s\),\s+each\s+with\s+a\s+value\s+of\s+(\d+)'
        for match in re.finditer(item_pattern, text):
            count = match.group(1)
            item_name = match.group(2)
            value = match.group(3)
            items[f"{item_name}_cnt"] = count
            items[f"{item_name}_val"] = value

        # Also try the context: format
        context_match = re.search(r'context:\s*(.+?)(?:\n|$)', text)
        if context_match and not items:
            context_str = context_match.group(1)
            # Try standard format: book=3 book_value=1
            for item_match in re.finditer(r'(\w+)=(\d+)\s+(\w+)_value=(\d+)', context_str):
                item_name = item_match.group(1)
                quantity = item_match.group(2)
                value = item_match.group(4)
                items[f"{item_name}_cnt"] = quantity
                items[f"{item_name}_val"] = value

            # Also check for _cnt/_val format from raw data
            if not items:
                for match in re.finditer(r'(\w+)_(cnt|val)=(\d+)', context_str):
                    item_base = match.group(1)
                    metric_type = match.group(2)
                    value = match.group(3)
                    items[f"{item_base}_{metric_type}"] = value

        state['items'] = items

        # Extract dialogue section - look for both formats
        dialogue_text = ""
        has_dialogue_section = False

        # Format 1: dialogue: marker
        dialogue_match = re.search(r'dialogue:\s*(.+)', text, re.DOTALL)
        if dialogue_match:
            dialogue_text = dialogue_match.group(1)
            has_dialogue_section = True
        # Format 2: <dialogue> tags
        elif '<dialogue>' in text:
            dialogue_match = re.search(r'<dialogue>\s*(.+?)\s*</dialogue>', text, re.DOTALL)
            if dialogue_match:
                dialogue_text = dialogue_match.group(1)
                has_dialogue_section = True
        else:
            # No dialogue section found - this might be a single-turn example
            dialogue_text = text
            has_dialogue_section = False

        # Parse dialogue turns - handle multiple formats
        # Deal or No Deal can have dialogue in different formats:
        # 1. YOU: message\nTHEM: message\nYOU: message
        # 2. YOU: message<eos>THEM: message<eos>YOU: message
        # 3. Mixed with <selection> tags that should be ignored

        # Split into turns preserving multi-line messages
        turns_raw = []

        if '<eos>' in dialogue_text:
            # Split by <eos> tags
            parts = dialogue_text.split('<eos>')
            for part in parts:
                part = part.strip()
                if part and part != '<selection>':
                    turns_raw.append(part)
        else:
            # Split by lines but group messages by speaker
            current_speaker = None
            current_message = []

            for line in dialogue_text.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Check if this line starts with a speaker label
                if line.startswith('YOU:') or line.startswith('THEM:'):
                    # Save previous turn if it exists
                    if current_speaker and current_message:
                        full_message = ' '.join(current_message)
                        turns_raw.append(f"{current_speaker}: {full_message}")

                    # Start new turn
                    if line.startswith('YOU:'):
                        current_speaker = 'YOU'
                        current_message = [line[4:].strip()]
                    else:  # THEM:
                        current_speaker = 'THEM'
                        current_message = [line[5:].strip()]
                else:
                    # Continuation of current message
                    if current_message:
                        current_message.append(line)

            # Don't forget last turn
            if current_speaker and current_message:
                full_message = ' '.join(current_message)
                turns_raw.append(f"{current_speaker}: {full_message}")

        # Process turns into structured dialogue
        if has_dialogue_section and turns_raw:
            # We have a full conversation - process all turns
            for i, turn_text in enumerate(turns_raw):
                turn_text = turn_text.strip()
                if not turn_text or turn_text == '<selection>':
                    continue

                # Determine speaker and extract content
                content = None
                side = None

                if turn_text.startswith('YOU:'):
                    content = turn_text[4:].strip()
                    side = 'agent'
                elif turn_text.startswith('THEM:'):
                    content = turn_text[5:].strip()
                    side = 'partner'
                else:
                    # No speaker label - could be a continuation or standalone
                    # Try to infer based on position
                    content = turn_text
                    side = 'agent' if len(dialogue) % 2 == 0 else 'partner'

                # Sanitize the content
                content = self._sanitize_dialogue_text(content)

                # Validate and add turn
                if content and self._is_valid_negotiation_text(content):
                    turn = {
                        'text': content,  # Clean text without speaker labels
                        'side': side,
                        'action': 'PROPOSE'
                    }
                    dialogue.append(turn)

            # Link responses properly - each turn should have a response from the next turn
            for i in range(len(dialogue) - 1):
                if dialogue[i]['side'] != dialogue[i+1]['side']:
                    dialogue[i]['response'] = dialogue[i+1]['text']

            # If we have labels, add them as the final response
            if labels and dialogue:
                for label in labels:
                    if not label:
                        continue
                    clean_label = self._sanitize_dialogue_text(label)
                    if self._is_valid_negotiation_text(clean_label):
                        # Add as response to last turn if it doesn't have one
                        if 'response' not in dialogue[-1]:
                            dialogue[-1]['response'] = clean_label
                        break

        else:
            # Single-turn example or no dialogue section
            # For Deal or No Deal, the text contains the context and previous message
            # Labels contain the agent's response
            if labels:
                for label in labels:
                    if not label or label == '<selection>':
                        continue

                    clean_label = self._sanitize_dialogue_text(label)
                    if clean_label:  # Accept any non-empty label for single-turn
                        # Extract the partner's last message from text
                        partner_text = ''

                        # Remove the initial context about items if present
                        clean_text = text
                        if 'There are' in clean_text and 'each with a value' in clean_text:
                            # Remove the initial negotiation setup text
                            parts = clean_text.split('\n', 1)
                            if len(parts) > 1:
                                clean_text = parts[1].strip()
                            else:
                                # The whole text might be just context, use empty partner message
                                clean_text = ''

                        # The remaining text is the partner's message
                        if clean_text:
                            partner_text = self._sanitize_dialogue_text(clean_text)

                        # Create a single dialogue turn with proper structure
                        dialogue.append({
                            'text': partner_text,  # Partner's last message
                            'side': 'partner',
                            'action': 'PROPOSE',
                            'response': clean_label  # Agent's response
                        })
                        break

        # For single-turn training examples, we just need one turn with a response
        # Don't require both participants for single-turn examples
        if len(dialogue) == 0:
            return [], state

        return dialogue, state

    def _is_valid_negotiation_text(self, text: str) -> bool:
        """
        Check if text is valid negotiation dialogue (not prompts or metadata).

        Args:
            text: Text to validate

        Returns:
            True if text appears to be actual negotiation dialogue
        """
        if not text or len(text) < 5:
            return False

        # Reject if contains obvious prompt/metadata markers
        reject_patterns = [
            r'^\s*##',  # Markdown headers
            r'you are a',  # Prompt instructions
            r'your target',  # Metadata
            r'make a negotiation',  # Instructions
            r'what is the',  # Questions about metadata
            r'^\s*\d+\.\s',  # Numbered lists
            r'^\s*\*\s',  # Bullet points
            r'subject:',  # Email-like headers
            r'dear seller',  # Formal letter openings
        ]

        text_lower = text.lower()
        for pattern in reject_patterns:
            if re.search(pattern, text_lower):
                return False

        # Accept if contains negotiation-related content
        negotiation_keywords = [
            'deal', 'offer', 'accept', 'price', 'want', 'need',
            'how about', 'would you', 'i can', "let's", 'agree',
            'works for me', 'sounds good', 'take', 'give', 'split'
        ]

        return any(keyword in text_lower for keyword in negotiation_keywords)
    
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

        # For dealnodeal, use specialized parser
        if task_name == 'dealnodeal':
            # Combine all text and labels from episode data
            combined_text = []
            all_labels = []

            for turn in episode_data:
                text = turn.get('text', '')
                labels = turn.get('labels', [])
                if text:
                    combined_text.append(text)
                if labels:
                    all_labels.extend(labels)

            # Parse using dealnodeal parser
            full_text = '\n'.join(combined_text)
            dialogue, parsed_state = self._parse_dealnodeal(full_text, all_labels)

            # Merge states
            episode_state = {'task': task_name, 'category': 'negotiation'}
            episode_state.update(state)
            episode_state.update(parsed_state)

            if not dialogue:
                return None

            episode = {
                'state': episode_state,
                'dialogue': dialogue,
                'meta': {
                    'source_task': task_name,
                    'episode_id': f"{task_name}_{abs(hash(full_text))%100000}",
                    'datatype': episode_data[0].get('datatype', 'train'),
                    'num_turns': len(dialogue)
                }
            }

            return episode

        # Generic handling for non-dealnodeal tasks
        dialogue = []
        full_text = []

        for i, turn in enumerate(episode_data):
            text = turn.get('text', '')
            labels = turn.get('labels', [])

            if text:
                # Sanitize text for all tasks
                text = self._sanitize_dialogue_text(text)
                full_text.append(text)

            # Build dialogue turn
            if labels:
                # Sanitize labels
                clean_labels = [self._sanitize_dialogue_text(label) for label in labels]
                valid_label = None

                # Find first valid label
                for label in clean_labels:
                    if self._is_valid_negotiation_text(label):
                        valid_label = label
                        break

                if valid_label:
                    dialogue.append({
                        'text': text,
                        'response': valid_label,
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