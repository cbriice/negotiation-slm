import json
from utils.protocol_validator import validate_episode_for_training

# Load first episode
with open('data/training/builtin_tasks/dealnodeal_episodes.jsonl', 'r') as f:
    episode = json.loads(f.readline())

# Test validation
is_suitable, reasons = validate_episode_for_training(episode)
print(f"Episode suitable: {is_suitable}")
print(f"Reasons: {reasons}")
print(f"\nEpisode keys: {episode.keys()}")
print(f"Dialogue length: {len(episode.get('dialogue', []))}")
print(f"Has negotiation_history: {'negotiation_history' in episode}")
print(f"Has metrics: {'metrics' in episode}")
