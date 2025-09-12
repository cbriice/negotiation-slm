"""
Test integration between EpisodeLogger and EpisodeMetrics.
Ensures that episode formats are compatible.
"""

import json
import tempfile
from pathlib import Path
import sys
sys.path.append('..')

from utils.episode_logger import EpisodeLogger, TrainingStage
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
                'logistics': {
                    'region': 'Bay Area',
                    'meet_ok': True,
                    'ship_ok': False
                }
            },
            'seller_state': {
                'min_price': 800,
                'target_price': 950,
                'logistics': {
                    'region': 'Bay Area',
                    'meet_ok': True,
                    'ship_ok': True,
                    'return_days': 7
                }
            }
        },
        'meta': {
            'episode_id': 'ep_test_001',
            'dialogue_id': 'dlg_test_001',
            'side': 'seller',
            'seed': 'test_seed_123'
        }
    }


def test_raw_episode_compatibility():
    """Test that raw episodes work with EpisodeMetrics."""
    print("Testing raw episode compatibility...")
    
    try:
        # Create sample episode
        episode = create_sample_episode()
        
        # Try to import and use EpisodeMetrics
        try:
            from evaluation.episode_metrics import EpisodeMetrics
            metrics_calc = EpisodeMetrics()
            metrics = metrics_calc.calculate(episode)
            
            # Verify key metrics are calculated
            assert 'episode_id' in metrics
            assert metrics['episode_id'] == 'ep_test_001'
            assert 'turns' in metrics
            assert metrics['turns'] == 4
            assert 'deal_closed' in metrics
            assert metrics['deal_closed'] == True  # Should detect COMMIT->ACK pattern
            assert 'final_price' in metrics
            assert metrics['final_price'] == 850.0
            
            print("✓ Raw episodes are compatible with EpisodeMetrics")
            return True
            
        except ImportError as e:
            print(f"  ⚠ Skipping EpisodeMetrics test (missing dependency: {e})")
            # Still verify episode structure is correct
            assert 'dialogue' in episode
            assert 'state' in episode
            assert 'meta' in episode
            assert len(episode['dialogue']) == 4
            print("✓ Raw episode structure is correct (metrics calculation skipped)")
            return True
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logged_episode_compatibility():
    """Test that episodes logged through EpisodeLogger work with EpisodeMetrics."""
    print("Testing logged episode compatibility...")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create episode logger
            logger = EpisodeLogger(
                base_dir=tmpdir,
                batch_size=1,
                stage=TrainingStage.SELFCHAT
            )
            
            # Log episode
            episode = create_sample_episode()
            logger.log_episode(episode, format='raw')
            logger.flush()
            
            # Read back the logged episode
            stage_dir = Path(tmpdir) / 'selfchat'
            logged_files = list(stage_dir.glob('episodes_*.jsonl'))
            assert len(logged_files) == 1
            
            # Load and verify
            with open(logged_files[0], 'r') as f:
                loaded_episode = json.loads(f.readline())
            
            # The loaded episode should have the same structure
            assert 'dialogue' in loaded_episode
            assert 'state' in loaded_episode
            assert 'meta' in loaded_episode
            
            # Should have added logging metadata
            assert 'logged_at' in loaded_episode['meta']
            assert loaded_episode['meta']['stage'] == 'selfchat'
            
            # Try to test with EpisodeMetrics if available
            try:
                from evaluation.episode_metrics import EpisodeMetrics
                metrics_calc = EpisodeMetrics()
                metrics = metrics_calc.calculate(loaded_episode)
                
                # Verify metrics still work
                assert metrics['episode_id'] == 'ep_test_001'
                assert metrics['turns'] == 4
                assert metrics['deal_closed'] == True
                
                print("✓ Logged episodes maintain compatibility with EpisodeMetrics")
            except ImportError:
                print("✓ Logged episode structure is correct (metrics calculation skipped)")
            
            return True
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sft_packet_structure():
    """Test that SFT packets have the correct structure."""
    print("Testing SFT packet structure...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EpisodeLogger(base_dir=tmpdir, batch_size=10)
        
        # Create SFT packets
        episode = create_sample_episode()
        packets = logger.transform_to_sft_packet(episode)
        
        # Verify packet structure matches schema
        for packet in packets:
            assert 'input' in packet
            assert 'output' in packet
            assert 'meta' in packet
            
            # Check input structure
            assert 'state' in packet['input']
            assert 'recent_convo' in packet['input']
            assert 'partner_msg' in packet['input']
            
            # Check output is a string
            assert isinstance(packet['output'], str)
            
            # Check meta has required fields
            assert 'episode_id' in packet['meta']
            assert 'turn' in packet['meta']
            assert 'side' in packet['meta']
        
        print(f"✓ SFT packets have correct structure ({len(packets)} packets created)")
        return True


def test_dpo_pair_structure():
    """Test that DPO pairs have the correct structure."""
    print("Testing DPO pair structure...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EpisodeLogger(base_dir=tmpdir)
        
        # Create two episodes with diverging outcomes
        chosen_episode = create_sample_episode()
        rejected_episode = create_sample_episode()
        
        # Modify rejected episode to have different outcome
        rejected_episode['dialogue'][2]['response'] = "[NO_DEAL] That's too low, I can't accept."
        rejected_episode['dialogue'][3] = {
            'side': 'seller',
            'text': "[NO_DEAL] That's too low, I can't accept.",
            'response': "Understood, thanks anyway."
        }
        
        # Create DPO pair
        dpo_pair = logger.transform_to_dpo_pair(chosen_episode, rejected_episode)
        
        # Verify structure matches schema
        assert dpo_pair is not None
        assert 'context' in dpo_pair
        assert 'chosen' in dpo_pair
        assert 'rejected' in dpo_pair
        assert 'meta' in dpo_pair
        
        # Check context structure
        assert 'state' in dpo_pair['context']
        assert 'recent_convo' in dpo_pair['context']
        assert 'partner_msg' in dpo_pair['context']
        
        # Check divergence was found correctly
        assert dpo_pair['chosen'] != dpo_pair['rejected']
        
        print("✓ DPO pairs have correct structure")
        return True


def test_streaming_compatibility():
    """Test that streamed episodes work with EpisodeMetrics."""
    print("Testing streaming compatibility...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EpisodeLogger(base_dir=tmpdir, batch_size=2)
        
        # Log multiple episodes
        for i in range(3):
            episode = create_sample_episode()
            episode['meta']['episode_id'] = f'ep_test_{i:03d}'
            logger.log_episode(episode, format='raw')
        logger.flush()
        
        # Stream episodes and calculate metrics
        metrics_calc = EpisodeMetrics()
        stage_dir = Path(tmpdir) / 'general'
        
        for filepath in stage_dir.glob('episodes_*.jsonl'):
            episode_count = 0
            for episode in logger.stream_episodes(filepath):
                metrics = metrics_calc.calculate(episode)
                assert 'episode_id' in metrics
                assert 'deal_closed' in metrics
                episode_count += 1
            
            print(f"  ✓ Streamed {episode_count} episodes from {filepath.name}")
        
        print("✓ Streaming maintains compatibility with EpisodeMetrics")
        return True


if __name__ == '__main__':
    print("=" * 60)
    print("Testing EpisodeLogger and EpisodeMetrics Integration")
    print("=" * 60)
    
    tests = [
        test_raw_episode_compatibility,
        test_logged_episode_compatibility,
        test_sft_packet_structure,
        test_dpo_pair_structure,
        test_streaming_compatibility
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✅ All integration tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)