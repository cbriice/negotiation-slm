from pathlib import Path
from typing import Dict, Any, Optional

BASE_DIR = Path(__file__).parent.parent

PARLAI_CONFIG = {
    'model': 'agents.ollama_wrapper:OllamaNegotiator',
    'task': 'tasks.negotiation_teacher:NegotiationTeacher',
    'datapath': str(BASE_DIR / 'data'),
    'batchsize': 1,  # Start with 1 for debugging, increase for production
    'max_turns': 20,
    'truncate': 1536,  # Match MAX_SEQ_LEN from training_parameters.py
    'history_size': 5,  # Number of previous turns to include in context
    'temperature': 0.7,
    'top_p': 0.9,
    'max_new_tokens': 256,
}

SELFCHAT_CONFIG = {
    'num_self_chats': 1500,  # From negotiation training plan step 3
    'selfchat_max_turns': 20,
    'seed_rotation': True,  # Rotate through different seed states
    'log_every_n_episodes': 100,
}

DIALOGPARTNER_CONFIG = {
    'num_episodes': 2000,  # From negotiation training plan step 6
    'buyer_seller_split': 0.5,  # 50/50 split on sides
    'rule_agent_personas': [
        'aggressive_negotiator',
        'cooperative_negotiator', 
        'analytical_negotiator',
        'emotional_negotiator',
        'time_pressured_negotiator',
    ],
    'enable_termination': True,  # RuleAgent can terminate on violations
}

BUILTIN_TASKS_CONFIG = {
    'dealnodeal': {
        'task': 'parlai.tasks.dealnodeal.agents:SystemTeacher',
        'datatype': 'train',
        'max_examples': -1,  # Use all available
        'export_format': 'sft',
    },
    'casino': {
        'task': 'parlai.tasks.casino.agents:SystemTeacher',
        'datatype': 'train', 
        'max_examples': -1,
        'export_format': 'sft',
    },
}

def get_parlai_opts(stage: str = 'baseline', **overrides) -> Dict[str, Any]:
    """Get ParlAI options for a specific training stage."""
    opts = PARLAI_CONFIG.copy()
    
    if stage == 'selfchat':
        opts.update(SELFCHAT_CONFIG)
    elif stage == 'dialogpartner':
        opts.update(DIALOGPARTNER_CONFIG)
    elif stage == 'builtin':
        opts['builtin_tasks'] = BUILTIN_TASKS_CONFIG
    
    # Apply any overrides
    opts.update(overrides)
    
    return opts

def get_log_path(stage: str, episode_id: Optional[str] = None) -> Path:
    """Get the log file path for a given stage and episode."""
    log_dir = BASE_DIR / 'data' / 'logs' / stage
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if episode_id:
        return log_dir / f'{episode_id}.jsonl'
    else:
        return log_dir