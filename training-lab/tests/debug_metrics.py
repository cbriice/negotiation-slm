"""
Debug script to understand why deal_closed detection is failing.
"""

import json
import sys
sys.path.append('..')

from evaluation.episode_metrics import EpisodeMetrics


def create_sample_episode():
    """Create a sample episode for testing."""
    return {
        'dialogue': [
            {
                'side': 'buyer',
                'text': "Hi, I'm interested in the laptop. Would you take $800?",
                'response': "[PROPOSE] That's below my asking price. How about $950?",
            },
            {
                'side': 'seller',
                'text': "[PROPOSE] That's below my asking price. How about $950?",
                'response': "I can go up to $850, that's my best offer.",
            },
            {
                'side': 'buyer',
                'text': "I can go up to $850, that's my best offer.",
                'response': '[COMMIT] I accept $850. {"price": 850, "dialogue_id": "test123", "status": "agreed", "side_signed": "seller"}',
            },
            {
                'side': 'seller',
                'text': '[COMMIT] I accept $850. {"price": 850, "dialogue_id": "test123", "status": "agreed", "side_signed": "seller"}',
                'response': '[ACK] Deal confirmed at $850. {"price": 850, "dialogue_id": "test123", "status": "agreed", "side_signed": "buyer"}',
            }
        ],
        'state': {
            'item': {
                'name': 'Laptop',
                'brand': 'Dell',
                'model': 'XPS 13',
                'condition': 'excellent',
                'list_price': 1000
            },
            'buyer_state': {
                'max_price': 900,
                'target_price': 800,
            },
            'seller_state': {
                'min_price': 800,
                'target_price': 950,
            }
        },
        'meta': {
            'episode_id': 'ep_test_001',
            'dialogue_id': 'dlg_test_001',
            'side': 'seller',
            'seed': 'test_seed_123'
        }
    }


def debug_deal_analysis():
    """Debug the deal analysis to see what's happening."""
    episode = create_sample_episode()
    metrics_calc = EpisodeMetrics()
    
    # Manually run the _analyze_deal_outcome method
    dialogue = episode.get('dialogue', [])
    
    print("Analyzing dialogue turns:")
    print("=" * 60)
    
    commit_found = False
    for i, turn in enumerate(dialogue):
        text = turn.get('text', '') + ' ' + turn.get('response', '')
        print(f"\nTurn {i}:")
        print(f"  Combined text: {text[:100]}...")
        print(f"  Has [COMMIT]: {'[COMMIT]' in text}")
        print(f"  Has [ACK]: {'[ACK]' in text}")
        
        if '[COMMIT]' in text:
            commit_found = True
            print(f"  -> COMMIT found! commit_found = {commit_found}")
            
            # Try to extract JSON
            first_brace = text.find('{')
            last_brace = text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                json_str = text[first_brace:last_brace + 1]
                print(f"  -> Extracted JSON: {json_str}")
                try:
                    terms = json.loads(json_str)
                    print(f"  -> Parsed terms: {terms}")
                except json.JSONDecodeError as e:
                    print(f"  -> JSON parse error: {e}")
        
        elif commit_found and '[ACK]' in text:
            print(f"  -> ACK found after COMMIT! Deal should be closed.")
            
            # Try to extract JSON
            first_brace = text.find('{')
            last_brace = text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                json_str = text[first_brace:last_brace + 1]
                print(f"  -> Extracted JSON: {json_str}")
                try:
                    terms = json.loads(json_str)
                    print(f"  -> Parsed terms: {terms}")
                except json.JSONDecodeError as e:
                    print(f"  -> JSON parse error: {e}")
    
    print("\n" + "=" * 60)
    print("Running full metrics calculation:")
    
    # Run the full calculation
    result = metrics_calc._analyze_deal_outcome(dialogue)
    print(f"Result: {result}")
    
    # Also run full metrics
    metrics = metrics_calc.calculate(episode)
    print(f"\nFull metrics deal_closed: {metrics.get('deal_closed')}")
    print(f"Final price: {metrics.get('final_price')}")


if __name__ == '__main__':
    debug_deal_analysis()