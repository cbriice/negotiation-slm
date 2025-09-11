#!/usr/bin/env python3
"""
Test script for DialogPartnerWorld with comprehensive logging.
Run this to test if DialogPartnerWorld is working properly.
"""

import sys
import logging
from pathlib import Path
from parlai.core.params import ParlaiParser
from worlds.dialog_world import DialogPartnerWorld

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)

# Set up comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('logs/agent_logs.log', mode='a')
    ],
    force=True
)

logger = logging.getLogger(__name__)

def main():
    """Run a simple DialogPartnerWorld test."""
    logger.info("=" * 60)
    logger.info("Starting DialogPartnerWorld test")
    logger.info("=" * 60)
    
    # Set up parser with DialogPartnerWorld arguments
    parser = ParlaiParser()
    DialogPartnerWorld.add_cmdline_args(parser)
    
    # Parse arguments (you can override these via command line)
    opt = parser.parse_args(args=[
        '--partner-mode', 'rule',
        '--num-episodes', '1',  # Just run 1 episode for testing
        '--max-turns', '10',
        '--temperature', '0.5',
        '--use-two-part-response', 'true',
        '--output-dir', 'data/test_dialog/',
        '--save-metrics-every', '1',
    ])
    
    logger.info(f"Parsed options: {opt}")
    
    try:
        # Create the world
        logger.info("Creating DialogPartnerWorld...")
        world = DialogPartnerWorld(opt)
        logger.info("DialogPartnerWorld created successfully")
        
        # Run episodes
        logger.info(f"Running {opt['num_episodes']} episode(s)...")
        episode_count = 0
        while episode_count < opt['num_episodes']:
            logger.info(f"\n{'='*40}")
            logger.info(f"Starting episode {episode_count + 1}")
            logger.info(f"{'='*40}")
            
            # Call parley to run one episode
            world.parley()
            episode_count += 1
            
            logger.info(f"Episode {episode_count} completed")
            logger.info(f"Total episodes run: {world.episode_count}")
        
        # Shutdown and save
        logger.info("\nShutting down world...")
        world.shutdown()
        logger.info("World shutdown complete")
        
        # Report results
        logger.info("\n" + "=" * 60)
        logger.info("TEST COMPLETED SUCCESSFULLY")
        logger.info(f"Episodes completed: {world.episode_count}")
        logger.info(f"Output directory: {world.output_dir}")
        logger.info("Check logs/agent_logs.log for detailed logs")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())