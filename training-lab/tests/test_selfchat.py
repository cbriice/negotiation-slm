#!/usr/bin/env python
"""
Test script for SelfChatWorld implementation.
Runs a small batch of episodes to verify functionality.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from parlai.core.params import ParlaiParser
from worlds.selfchat_world import SelfChatWorld
from agents.ollama_wrapper import OllamaNegotiator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_selfchat():
    """Test SelfChatWorld with a small batch of episodes."""
    
    # Create parser and add arguments
    parser = ParlaiParser(False, False)
    SelfChatWorld.add_cmdline_args(parser)
    OllamaNegotiator.add_cmdline_args(parser)
    
    # Parse with test configuration
    opt = parser.parse_args([
        '--num-episodes', '5',  # Small test batch
        '--batch-size', '2',
        '--max-turns', '10',
        '--output-dir', 'data/test_selfchat/',
        '--model-name', 'mistral:7b',
        '--temperature', '0.7',
        '--ollama-host', 'http://localhost:11434',
        '--save-metrics-every', '2'
    ])
    
    # Add required fields for ParlAI
    opt['task'] = 'selfplay'
    opt['datatype'] = 'train'  # Required by ParlAI Teacher
    
    logger.info("="*50)
    logger.info("TESTING SELFCHAT WORLD IMPLEMENTATION")
    logger.info("="*50)
    logger.info(f"Running {opt['num_episodes']} test episodes")
    logger.info(f"Output to: {opt['output_dir']}")
    logger.info("="*50)
    
    try:
        # Create world
        logger.info("\n1. Creating SelfChatWorld...")
        world = SelfChatWorld(opt)
        logger.info(f"   ✓ World created with {len(world.state_seeds)} seeds")
        
        # Load seeds
        logger.info("\n2. Loading state seeds...")
        seeds, regression = world.load_state_seeds()
        logger.info(f"   ✓ Loaded {len(seeds)} active seeds")
        logger.info(f"   ✓ Loaded {len(regression)} regression seeds")
        
        # Run test episodes
        logger.info("\n3. Running test episodes...")
        episode_count = 0
        while not world.episode_done():
            world.parley()
            episode_count += 1
            logger.info(f"   Episode {episode_count} complete")
        
        # Check generated data
        logger.info("\n4. Verifying generated data...")
        
        # Check episode files
        episode_files = list(world.episodes_dir.glob("*.jsonl"))
        logger.info(f"   ✓ Generated {len(episode_files)} episode files")
        
        # Check metrics
        metrics_files = list(world.metrics_dir.glob("*.csv"))
        logger.info(f"   ✓ Generated {len(metrics_files)} metrics files")
        
        # Shutdown
        logger.info("\n5. Shutting down and saving final data...")
        world.shutdown()
        logger.info("   ✓ Shutdown complete")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("TEST COMPLETED SUCCESSFULLY")
        logger.info("="*50)
        logger.info(f"✓ Generated {episode_count} episodes")
        logger.info(f"✓ Output saved to {opt['output_dir']}")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = test_selfchat()
    sys.exit(0 if success else 1)