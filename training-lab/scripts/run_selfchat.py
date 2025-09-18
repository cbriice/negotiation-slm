#!/usr/bin/env python
"""
Main script to run SelfChatWorld simulations for negotiation training.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
from worlds.selfchat_world import SelfChatWorld

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelfChatRunner(ParlaiScript):
    """Runner script for SelfChatWorld simulations."""
    
    @classmethod
    def setup_args(cls):
        """Set up command-line arguments."""
        parser = ParlaiParser(
            True, 
            True,
            description='Run self-chat negotiation simulations'
        )
        
        # Add SelfChatWorld arguments
        SelfChatWorld.add_cmdline_args(parser)
        
        # Add required task argument
        parser.add_argument(
            '--task',
            type=str,
            default='selfplay',
            help='Task name (required by ParlAI)'
        )
        
        # Add model arguments
        parser.add_argument(
            '--model',
            type=str,
            default='agents.ollama_wrapper:OllamaNegotiator',
            help='Model to use for agents'
        )
        
        parser.add_argument(
            '--model-name',
            type=str,
            default='mistral:7b',
            help='Ollama model name'
        )
        
        parser.add_argument(
            '--use-gpu',
            type=bool,
            default=True,
            help='Use GPU if available'
        )
        
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug logging'
        )

        parser.add_argument(
            '--use-two-part-response',
            type=lambda x: x.lower() in ['true', '1', 'yes'],
            default=True,
            help='Enable two-part response generation (strategy then response). Accepts: true/false, 1/0, yes/no'
        )

        return parser
    
    def run(self):
        """Run the self-chat simulations."""
        opt = self.opt
        
        # Ensure task is set for ParlAI World
        if 'task' not in opt:
            opt['task'] = 'selfplay'
        
        # Set up debug logging if requested
        if opt.get('debug'):
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info("="*60)
        logger.info("STARTING SELFCHAT NEGOTIATION SIMULATIONS")
        logger.info("="*60)
        logger.info(f"Target episodes: {opt.get('num_episodes', 1500)}")
        logger.info(f"Max turns per episode: {opt.get('max_turns', 20)}")
        logger.info(f"Output directory: {opt.get('output_dir', 'data/negotiations/selfchat/')}")
        logger.info(f"Model: {opt.get('model_name', 'mistral:7b')}")
        if opt.get('model_checkpoint'):
            logger.info(f"LoRA checkpoint: {opt.get('model_checkpoint')}")
        logger.info(f"Two-part response: {opt.get('use_two_part_response', True)}")
        logger.info("="*60 + "\n")
        
        # Initialize world variable
        world = None
        
        try:
            # Create SelfChatWorld
            world = SelfChatWorld(opt)
            
            # Load seeds
            world.load_state_seeds()
            
            # Run simulations
            logger.info("Starting episode generation...")
            while not world.episode_done():
                world.parley()
            
            # Shutdown and save final data
            logger.info("Episode generation complete. Saving final data...")
            world.shutdown()
            
            logger.info("\n" + "="*60)
            logger.info("SELFCHAT SIMULATIONS COMPLETE")
            logger.info("="*60)
            
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user. Saving current progress...")
            if world is not None:
                world.shutdown()
            logger.info("Progress saved. Exiting.")
        
        except Exception as e:
            logger.error(f"Error during simulation: {e}", exc_info=True)
            if world is not None:
                logger.info("Attempting to save current progress...")
                try:
                    world.shutdown()
                    logger.info("Progress saved despite error.")
                except:
                    logger.error("Could not save progress.")
            raise


def main():
    """Main entry point."""
    SelfChatRunner.main()


if __name__ == '__main__':
    main()