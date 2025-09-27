#!/usr/bin/env python
"""
Main script to run SelfChatWorld simulations for negotiation training.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
from worlds.selfchat_world import SelfChatWorld
from config.training_config import get_latest_adapter_path

# Set up logging with DEBUG level and detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for comprehensive logging
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('selfchat_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Ensure all loggers allow DEBUG messages through
logging.getLogger('worlds.selfchat_world').setLevel(logging.DEBUG)
logging.getLogger('agents.ollama_wrapper').setLevel(logging.DEBUG)
logging.getLogger('agents.negotiator_base').setLevel(logging.DEBUG)
logging.getLogger('tasks.selfplay_teacher').setLevel(logging.DEBUG)


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
            default='agents.hf_negotiator:HFNegotiator',
            help='Model to use for agents'
        )

        parser.add_argument(
            '--model-name',
            type=str,
            default='mistralai/Mistral-7B-v0.1',
            help='HuggingFace model name'
        )

        parser.add_argument(
            '--adapter-path',
            type=str,
            default=None,
            help='Path to LoRA adapter checkpoint'
        )

        parser.add_argument(
            '--use-latest-checkpoint',
            action='store_true',
            default=True,
            help='Automatically load the latest adapter checkpoint'
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

        # Auto-load latest checkpoint if requested and no adapter path provided
        if opt.get('use_latest_checkpoint') and not opt.get('adapter_path'):
            latest_adapter = get_latest_adapter_path()
            if latest_adapter:
                opt['adapter_path'] = str(latest_adapter)
                logger.info(f"Auto-loading latest adapter checkpoint: {latest_adapter}")
            else:
                logger.info("No adapter checkpoints found, using base model")
        
        # Set up debug logging if requested
        if opt.get('debug'):
            # Set root logger to DEBUG
            logging.getLogger().setLevel(logging.DEBUG)
            # Also set specific module loggers to DEBUG
            logging.getLogger('agents.ollama_wrapper').setLevel(logging.DEBUG)
            logging.getLogger('worlds.selfchat_world').setLevel(logging.DEBUG)
            logging.getLogger('tasks').setLevel(logging.DEBUG)
            # Update the console handler format to show more detail
            for handler in logging.getLogger().handlers:
                handler.setLevel(logging.DEBUG)
        
        logger.info("="*60)
        logger.info("STARTING SELFCHAT NEGOTIATION SIMULATIONS")
        logger.info("="*60)
        logger.info(f"Target episodes: {opt.get('num_episodes', 1500)}")
        logger.info(f"Max turns per episode: {opt.get('max_turns', 20)}")
        logger.info(f"Output directory: {opt.get('output_dir', 'data/negotiations/selfchat/')}")
        logger.info(f"Model: {opt.get('model_name', 'mistralai/Mistral-7B-v0.1')}")
        if opt.get('adapter_path'):
            logger.info(f"LoRA adapter: {opt.get('adapter_path')}")
        elif opt.get('model_checkpoint'):
            logger.info(f"LoRA checkpoint: {opt.get('model_checkpoint')}")
        logger.info(f"Two-part response: {opt.get('use_two_part_response', True)}")
        logger.info(f"Use latest checkpoint: {opt.get('use_latest_checkpoint', True)}")
        logger.info("="*60 + "\n")
        
        # Initialize world variable
        world = None
        
        try:
            # Create SelfChatWorld
            # Pass adapter path to world if available
            if opt.get('adapter_path'):
                logger.info(f"Initializing world with adapter: {opt['adapter_path']}")

            logger.debug("Creating SelfChatWorld instance...")
            world = SelfChatWorld(opt)
            logger.debug("SelfChatWorld instance created successfully")
            
            # Load seeds
            logger.debug("Loading state seeds...")
            seeds, regression = world.load_state_seeds()
            logger.debug(f"Loaded {len(seeds)} active seeds and {len(regression)} regression seeds")
            
            # Run simulations
            logger.info("Starting episode generation...")
            episode_num = 0
            while not world.episode_done():
                episode_num += 1
                logger.debug(f"\n{'='*50}")
                logger.debug(f"Starting parley for episode {episode_num}/{opt.get('num_episodes', 1500)}")
                logger.debug(f"Current episode count: {world.episode_count}")
                world.parley()
                logger.debug(f"Completed parley for episode {episode_num}")
                logger.debug(f"Episode done: {world.episode_done()}")
            
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