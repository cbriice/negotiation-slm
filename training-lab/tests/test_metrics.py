#!/usr/bin/env python3
"""
Test script for the metrics tracking system.
Creates sample negotiation episodes and tests all metrics components.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

from evaluation.episode_metrics import EpisodeMetrics
from evaluation.session_metrics import SessionMetrics
from evaluation.csv_logger import CSVLogger
from evaluation.metrics_integration import MetricsIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_episode(episode_id: str, deal_closed: bool = True, side: str = 'buyer'):
    """Create a sample negotiation episode for testing."""
    episode = {
        'meta': {
            'episode_id': episode_id,
            'dialogue_id': f'dialogue_{episode_id}',
            'side': side,
            'seed': 'test_seed_001'
        },
        'state': {
            'item': {
                'name': 'iPhone 12',
                'brand': 'Apple',
                'model': '12 Pro',
                'condition': 'Good',
                'list_price': 800
            },
            'buyer_state': {
                'max_price': 750,
                'target_price': 650,
                'preferences': {'color': 'blue', 'storage': '128GB'},
                'logistics': {
                    'region': 'Bay Area',
                    'meet_ok': True,
                    'ship_ok': False,
                    'return_days': 7
                }
            },
            'seller_state': {
                'min_price': 600,
                'target_price': 700,
                'item_issues': ['minor scratches'],
                'logistics': {
                    'region': 'Bay Area',
                    'meet_ok': True,
                    'ship_ok': True,
                    'return_days': 3
                }
            }
        },
        'dialogue': []
    }
    
    # Add dialogue turns based on whether deal closes
    if deal_closed:
        episode['dialogue'] = [
            {
                'side': 'buyer',
                'text': '[PROPOSE] Hi, I saw your iPhone 12 listing. Would you consider $600?',
                'response': ''
            },
            {
                'side': 'seller',
                'text': '',
                'response': '[PROPOSE] Thanks for your interest! Given the good condition and included accessories, I can go down to $700.'
            },
            {
                'side': 'buyer',
                'text': '[PROPOSE] I understand. Since there are minor scratches as you mentioned, how about $650?',
                'response': ''
            },
            {
                'side': 'seller',
                'text': '',
                'response': '[PROPOSE] That\'s reasonable considering the scratches. Let\'s meet in the middle at $675?'
            },
            {
                'side': 'buyer',
                'text': '[COMMIT] {"price": 675, "location": "downtown", "payment": "cash", "delivery": "in-person"}',
                'response': ''
            },
            {
                'side': 'seller',
                'text': '',
                'response': '[ACK] {"price": 675, "location": "downtown", "payment": "cash", "delivery": "in-person"}'
            }
        ]
    else:
        episode['dialogue'] = [
            {
                'side': 'buyer',
                'text': '[PROPOSE] Hi, would you take $500 for the iPhone?',
                'response': ''
            },
            {
                'side': 'seller',
                'text': '',
                'response': '[PROPOSE] That\'s too low for me. The lowest I can go is $700.'
            },
            {
                'side': 'buyer',
                'text': '[PROPOSE] I can\'t go above $550.',
                'response': ''
            },
            {
                'side': 'seller',
                'text': '',
                'response': '[NO_DEAL] Sorry, we\'re too far apart on price. Good luck with your search!'
            }
        ]
    
    return episode


def create_protocol_violation_episode():
    """Create an episode with protocol violations for testing."""
    return {
        'meta': {
            'episode_id': 'violation_test',
            'dialogue_id': 'dialogue_violation',
            'side': 'buyer',
            'seed': 'violation_seed'
        },
        'state': {
            'item': {'name': 'Test Item', 'list_price': 100},
            'buyer_state': {'max_price': 100, 'target_price': 80},
            'seller_state': {'min_price': 70, 'target_price': 90}
        },
        'dialogue': [
            {
                'side': 'buyer',
                'text': '[PROPOSE] [COMMIT] I want to buy it for $75',  # Multiple tags
                'response': ''
            },
            {
                'side': 'seller',
                'text': '',
                'response': '[ACK] OK'  # ACK without TERMS_JSON
            },
            {
                'side': 'buyer',
                'text': '[PROPOSE] Actually, let me change that',  # PROPOSE after COMMIT
                'response': ''
            },
            {
                'side': 'seller',
                'text': '',
                'response': '[NACK] No'  # NACK without explanation
            },
            {
                'side': 'buyer',
                'text': '[NO_DEAL] Never mind',  # Early NO_DEAL
                'response': ''
            }
        ]
    }


def test_episode_metrics():
    """Test the EpisodeMetrics class."""
    logger.info("Testing EpisodeMetrics...")
    
    metrics_calc = EpisodeMetrics()
    
    # Test successful deal
    episode1 = create_sample_episode('test_001', deal_closed=True)
    metrics1 = metrics_calc.calculate(episode1)
    
    assert metrics1['deal_closed'] == True
    assert metrics1['final_price'] == 675
    assert metrics1['turns'] == 6
    assert metrics1['num_proposals'] == 4
    assert metrics1['violation_count'] == 0
    logger.info(f"✓ Successful deal metrics: price=${metrics1['final_price']}, turns={metrics1['turns']}")
    
    # Test failed deal
    episode2 = create_sample_episode('test_002', deal_closed=False)
    metrics2 = metrics_calc.calculate(episode2)
    
    assert metrics2['deal_closed'] == False
    assert metrics2['final_price'] == 0
    assert metrics2['turns'] == 4
    logger.info(f"✓ Failed deal metrics: deal_closed={metrics2['deal_closed']}, turns={metrics2['turns']}")
    
    # Test protocol violations
    episode3 = create_protocol_violation_episode()
    metrics3 = metrics_calc.calculate(episode3)
    
    assert metrics3['violation_count'] > 0
    logger.info(f"✓ Protocol violations detected: {metrics3['violation_count']} violations")
    for violation in metrics3['protocol_violations']:
        logger.info(f"  - {violation}")
    
    return [metrics1, metrics2, metrics3]


def test_session_metrics(episode_metrics_list):
    """Test the SessionMetrics class."""
    logger.info("\nTesting SessionMetrics...")
    
    session_calc = SessionMetrics()
    session_metrics = session_calc.aggregate(episode_metrics_list)
    
    assert session_metrics['total_episodes'] == len(episode_metrics_list)
    assert 0 <= session_metrics['deals_closed_pct'] <= 100
    assert session_metrics['avg_turns'] > 0
    
    logger.info(f"✓ Session aggregated: {session_metrics['total_episodes']} episodes")
    logger.info(f"  - Deals closed: {session_metrics['deals_closed_pct']:.1f}%")
    logger.info(f"  - Avg turns: {session_metrics['avg_turns']:.1f}")
    logger.info(f"  - Avg violations: {session_metrics['avg_violations_per_episode']:.2f}")
    
    return session_metrics


def test_csv_logger():
    """Test the CSVLogger class."""
    logger.info("\nTesting CSVLogger...")
    
    # Create test directory
    test_dir = Path('data/test_metrics')
    csv_logger = CSVLogger(test_dir)
    
    # Test episode logging
    episode = create_sample_episode('csv_test_001', deal_closed=True)
    metrics_calc = EpisodeMetrics()
    metrics = metrics_calc.calculate(episode)
    
    csv_logger.log_episode(metrics, stage='test')
    logger.info(f"✓ Episode logged to CSV")
    
    # Test session logging
    session_calc = SessionMetrics()
    session_metrics = session_calc.aggregate([metrics])
    
    csv_logger.log_session(session_metrics, stage='test')
    logger.info(f"✓ Session logged to CSV")
    
    # Test reading back
    episodes_df = csv_logger.read_episodes(stage='test')
    assert len(episodes_df) > 0
    logger.info(f"✓ Read {len(episodes_df)} episodes from CSV")
    
    sessions_df = csv_logger.read_sessions(stage='test')
    assert len(sessions_df) > 0
    logger.info(f"✓ Read {len(sessions_df)} sessions from CSV")
    
    # Get summary
    summary = csv_logger.get_episode_summary(stage='test')
    logger.info(f"✓ Episode summary: {summary}")
    
    return csv_logger


def test_metrics_integration():
    """Test the MetricsIntegration class."""
    logger.info("\nTesting MetricsIntegration...")
    
    # Initialize integration
    integration = MetricsIntegration(data_dir='data/test_metrics')
    integration.set_stage('integration_test')
    
    # Track individual episodes
    episode1 = create_sample_episode('int_test_001', deal_closed=True, side='buyer')
    metrics1 = integration.track_episode(episode1)
    logger.info(f"✓ Tracked episode 1: {metrics1['episode_id']}")
    
    episode2 = create_sample_episode('int_test_002', deal_closed=False, side='seller')
    metrics2 = integration.track_episode(episode2)
    logger.info(f"✓ Tracked episode 2: {metrics2['episode_id']}")
    
    # Track batch
    batch = [
        create_sample_episode('batch_001', deal_closed=True, side='buyer'),
        create_sample_episode('batch_002', deal_closed=True, side='seller'),
        create_sample_episode('batch_003', deal_closed=False, side='buyer')
    ]
    batch_metrics = integration.track_batch(batch)
    logger.info(f"✓ Tracked batch of {len(batch)} episodes")
    
    # Get current stats
    stats = integration.get_current_stats()
    logger.info(f"✓ Current stats: {stats['total_episodes']} episodes, {stats['deals_closed_pct']:.1f}% closed")
    
    # Get violation summary
    violations = integration.get_violation_summary()
    logger.info(f"✓ Violation summary: {len(violations)} unique violation types")
    
    # Log summary (will go to logger, not console)
    integration.log_summary()
    
    # Export session data
    export_path = integration.export_session_data()
    logger.info(f"✓ Exported session data to {export_path}")
    
    # Finalize session
    final_metrics = integration.finalize_session()
    logger.info(f"✓ Session finalized: {final_metrics['session_id']}")
    
    return integration


def test_edge_cases():
    """Test edge cases and error handling."""
    logger.info("\nTesting edge cases...")
    
    metrics_calc = EpisodeMetrics()
    
    # Empty episode
    empty_episode = {
        'meta': {'episode_id': 'empty'},
        'state': {},
        'dialogue': []
    }
    metrics = metrics_calc.calculate(empty_episode)
    assert metrics['turns'] == 0
    assert metrics['deal_closed'] == False
    logger.info("✓ Handled empty episode")
    
    # Episode with missing fields
    partial_episode = {
        'meta': {'episode_id': 'partial'},
        'dialogue': [
            {'text': 'Hello', 'response': 'Hi'}
        ]
    }
    metrics = metrics_calc.calculate(partial_episode)
    assert metrics['episode_id'] == 'partial'
    logger.info("✓ Handled partial episode data")
    
    # Session with no episodes
    session_calc = SessionMetrics()
    empty_session = session_calc.aggregate([])
    assert empty_session['total_episodes'] == 0
    logger.info("✓ Handled empty session")


def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("METRICS TRACKING SYSTEM TEST SUITE")
    logger.info("="*60)
    
    try:
        # Test individual components
        episode_metrics = test_episode_metrics()
        session_metrics = test_session_metrics(episode_metrics)
        csv_logger = test_csv_logger()
        
        # Test integration
        integration = test_metrics_integration()
        
        # Test edge cases
        test_edge_cases()
        
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("="*60)
        
    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == '__main__':
    main()