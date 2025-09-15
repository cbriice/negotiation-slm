"""
DPO (Direct Preference Optimization) training module using TRL's DPOTrainer.
Handles QLoRA DPO training with reward margin tracking and early stopping.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
    AutoTokenizer
)
from trl import DPOTrainer as TRLDPOTrainer, DPOConfig

from peft.mapping import get_peft_model

from training.model_loader import ModelLoader
from training.lora_config import LoRAConfigurator
from training.checkpoint_manager import CheckpointManager
from training.metrics_tracker import TrainingMetricsTracker
import training_parameters as params

logger = logging.getLogger(__name__)


class DPODataset(Dataset):
    """Custom dataset for DPO training from NDJSON files."""

    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: Optional[int] = None):
        """
        Initialize DPO dataset.

        Args:
            data_path: Path to NDJSON file with DPO pairs
            tokenizer: Tokenizer for processing text
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length or params.MAX_SEQ_LEN
        self.max_prompt_length = params.DPO_MAX_PROMPT_LEN
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load DPO pairs from NDJSON file."""
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Validate required fields
                    if all(k in item for k in ['context', 'chosen', 'rejected']):
                        data.append(item)
                    else:
                        logger.warning(f"Skipping invalid DPO pair: missing required fields")

        logger.info(f"Loaded {len(data)} DPO pairs from {data_path}")
        return data

    def _format_prompt(self, context: Dict) -> str:
        """Format context into a prompt string."""
        state = context.get('state', {})
        recent_convo = context.get('recent_convo', [])
        partner_msg = context.get('partner_msg', '')

        # Build prompt
        prompt_parts = []

        # Add state information if present
        if state:
            state_str = json.dumps(state, indent=2)
            prompt_parts.append(f"Negotiation State:\n{state_str}")

        # Add conversation history
        if recent_convo:
            prompt_parts.append("\nRecent Conversation:")
            for turn in recent_convo:
                side = turn.get('side', 'unknown')
                text = turn.get('text', '')
                if text:
                    prompt_parts.append(f"{side}: {text}")
                response = turn.get('response', '')
                if response:
                    prompt_parts.append(f"You: {response}")

        # Add current partner message
        if partner_msg:
            prompt_parts.append(f"\nPartner: {partner_msg}")
            prompt_parts.append("You:")

        return "\n".join(prompt_parts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single DPO training example."""
        item = self.data[idx]

        # Format prompt from context
        prompt = self._format_prompt(item['context'])

        # Return in format expected by TRL's DPOTrainer
        return {
            'prompt': prompt,
            'chosen': item['chosen'],
            'rejected': item['rejected']
        }


class DPOTrainingModule:
    """
    DPO training module using TRL's DPOTrainer with QLoRA.
    Integrates with existing data pipeline from EpisodeLogger/DataConverter.
    """

    def __init__(self,
                 model_name: str = "mistralai/Mistral-7B-v0.1",
                 output_dir: str = "checkpoints/dpo",
                 reference_model_path: Optional[str] = None):
        """
        Initialize DPO training module.

        Args:
            model_name: Base model name or path
            output_dir: Directory for saving checkpoints
            reference_model_path: Path to reference model (if None, uses base model)
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reference_model_path = reference_model_path or model_name

        # Initialize components
        self.checkpoint_manager = CheckpointManager(str(self.output_dir))
        self.metrics_tracker = TrainingMetricsTracker(log_dir=str(self.output_dir / "metrics"), stage="dpo")

        # Track training state
        self.reward_margins = []
        self.best_reward_margin = float('-inf')
        self.patience_counter = 0
        self.early_stopping_patience = 5

    def prepare_dataset(self, data_path: str, tokenizer: AutoTokenizer) -> HFDataset:
        """
        Prepare dataset for DPO training.

        Args:
            data_path: Path to NDJSON file with DPO pairs
            tokenizer: Tokenizer for processing

        Returns:
            HuggingFace Dataset ready for DPOTrainer
        """
        # Load custom dataset
        dataset = DPODataset(data_path, tokenizer)

        # Convert to HuggingFace Dataset format
        data_dict = {
            'prompt': [],
            'chosen': [],
            'rejected': []
        }

        for i in range(len(dataset)):
            item = dataset[i]
            data_dict['prompt'].append(item['prompt'])
            data_dict['chosen'].append(item['chosen'])
            data_dict['rejected'].append(item['rejected'])

        hf_dataset = HFDataset.from_dict(data_dict)

        logger.info(f"Prepared {len(hf_dataset)} DPO pairs for training")
        return hf_dataset

    def setup_training_args(self, num_train_epochs: Optional[int] = None) -> DPOConfig:
        """
        Setup training arguments for DPO.

        Args:
            num_train_epochs: Number of training epochs (overrides max_steps if set)

        Returns:
            DPOConfig configured for DPO training
        """
        # When load_best_model_at_end is True, eval_strategy must match save_strategy
        # For DPO, we want to save and evaluate at the same intervals
        save_strategy = params.SAVE_STRATEGY
        eval_strategy = "steps" if save_strategy == "steps" else "epoch"
        eval_steps = params.SAVE_STEPS if save_strategy == "steps" else None

        args = DPOConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=num_train_epochs if num_train_epochs else -1,
            max_steps=params.DPO_MAX_STEPS if not num_train_epochs else -1,
            per_device_train_batch_size=params.PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=params.GRAD_ACCUM_STEPS,
            learning_rate=params.DPO_LR,
            optim=params.OPTIM,
            save_steps=params.SAVE_STEPS,
            save_strategy=save_strategy,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            logging_steps=params.LOGGING_STEPS,
            warmup_ratio=0.1,  # Fixed warmup for DPO
            lr_scheduler_type="linear",  # Linear decay for DPO
            bf16=params.BF16,
            gradient_checkpointing=params.GRADIENT_CHECKPOINTING,
            remove_unused_columns=False,
            # DPO-specific parameters
            beta=params.DPO_BETA,
            label_smoothing=params.DPO_LABEL_SMOOTH,
            max_length=params.MAX_SEQ_LEN,
            max_prompt_length=params.DPO_MAX_PROMPT_LEN,
            generate_during_eval=False,
            load_best_model_at_end=True,
            metric_for_best_model="reward_margin",
            greater_is_better=True,
            report_to=["tensorboard"],
            logging_dir=str(self.output_dir / "logs"),
            seed=params.SEED,
        )

        return args

    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute metrics for DPO training.

        Args:
            eval_pred: Evaluation predictions

        Returns:
            Dictionary of metrics
        """
        # Extract rewards from predictions
        predictions = eval_pred.predictions

        if len(predictions) >= 2:
            # Convert to numpy arrays if they're lists
            chosen_rewards = np.array(predictions[0])
            rejected_rewards = np.array(predictions[1])

            # Compute reward margin (higher is better)
            reward_margins = chosen_rewards - rejected_rewards
            avg_margin = float(np.mean(reward_margins))

            # Track for early stopping
            self.reward_margins.append(avg_margin)

            # Check for improvement
            if avg_margin > self.best_reward_margin:
                self.best_reward_margin = avg_margin
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            metrics = {
                'reward_margin': avg_margin,
                'chosen_reward': float(np.mean(chosen_rewards)),
                'rejected_reward': float(np.mean(rejected_rewards)),
                'patience': self.patience_counter
            }

            # Log to metrics tracker
            self.metrics_tracker.log_training_step(
                step=len(self.reward_margins),
                epoch=0.0,  # Will be updated by trainer
                metrics=metrics
            )

            return metrics

        return {'reward_margin': 0.0}

    def should_early_stop(self) -> bool:
        """
        Check if training should stop early based on reward divergence.

        Returns:
            True if should stop early
        """
        # Check patience
        if self.patience_counter >= self.early_stopping_patience:
            logger.warning(f"Early stopping triggered: no improvement for {self.patience_counter} evaluations")
            return True

        # Check for reward collapse (margins getting too small)
        if len(self.reward_margins) >= 5:
            recent_margins = self.reward_margins[-5:]
            avg_recent = sum(recent_margins) / len(recent_margins)
            if avg_recent < 0.01:  # Threshold for collapse
                logger.warning(f"Early stopping triggered: reward margin collapsed to {avg_recent}")
                return True

        return False

    def train(self,
              train_data_path: str,
              eval_data_path: Optional[str] = None,
              num_train_epochs: Optional[int] = None,
              resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Run DPO training.

        Args:
            train_data_path: Path to training DPO pairs (NDJSON)
            eval_data_path: Optional path to eval DPO pairs
            num_train_epochs: Number of epochs (overrides max_steps)
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Dictionary with training results and metrics
        """
        logger.info("Starting DPO training...")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Reference model: {self.reference_model_path}")
        logger.info(f"Training data: {train_data_path}")

        # Load model and tokenizer
        logger.info("Loading model with QLoRA...")
        model, tokenizer = ModelLoader.load_model(self.model_name)

        # Load reference model (frozen in 4-bit)
        logger.info("Loading reference model...")
        ref_model = ModelLoader.load_reference_model(self.reference_model_path)

        # Apply LoRA configuration
        lora_config = LoRAConfigurator.get_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_data_path, tokenizer)
        eval_dataset = None
        if eval_data_path:
            eval_dataset = self.prepare_dataset(eval_data_path, tokenizer)

        # Setup training arguments
        training_args = self.setup_training_args(num_train_epochs)

        # Initialize DPO trainer
        trainer = TRLDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics if eval_dataset else None,
        )

        # Add early stopping callback if we have eval data
        if eval_dataset:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_threshold=0.001
            )
            trainer.add_callback(early_stopping)

        # Start training
        logger.info("Starting DPO training loop...")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model
        logger.info("Saving final model...")
        trainer.save_model(str(self.output_dir / "final_model"))
        tokenizer.save_pretrained(str(self.output_dir / "final_model"))

        # Save training metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "reward_margins": self.reward_margins,
            "best_reward_margin": self.best_reward_margin,
            "total_steps": train_result.global_step,
        }

        # Save metrics to file
        with open(self.output_dir / "training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Log final statistics
        logger.info("=" * 50)
        logger.info("DPO Training Complete!")
        logger.info(f"Final loss: {metrics['train_loss']:.4f}")
        logger.info(f"Best reward margin: {self.best_reward_margin:.4f}")
        logger.info(f"Total steps: {metrics['total_steps']}")
        logger.info(f"Model saved to: {self.output_dir / 'final_model'}")
        logger.info("=" * 50)

        return metrics

    def train_from_episodes(self,
                            episodes_path: str,
                            min_score_gap: float = 0.3,
                            num_train_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train DPO directly from episode files by auto-generating pairs.

        Args:
            episodes_path: Path to episodes NDJSON file
            min_score_gap: Minimum protocol score gap for pair selection
            num_train_epochs: Number of training epochs

        Returns:
            Training results
        """
        from training.data_converter import DataConverter

        logger.info(f"Auto-generating DPO pairs from episodes: {episodes_path}")

        # Use DataConverter to generate DPO pairs
        converter = DataConverter(output_dir=str(self.output_dir / "data"))
        dpo_file = converter.auto_generate_dpo(
            episodes_path,
            output_file=str(self.output_dir / "data" / "auto_dpo_pairs.jsonl"),
            score_threshold=min_score_gap
        )

        # Train on generated pairs
        return self.train(
            train_data_path=dpo_file,
            num_train_epochs=num_train_epochs
        )

    def merge_and_save(self, checkpoint_path: str, output_path: str):
        """
        Merge LoRA weights with base model and save.

        Args:
            checkpoint_path: Path to LoRA checkpoint
            output_path: Path to save merged model
        """
        from peft.peft_model import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Merging LoRA weights from {checkpoint_path}...")

        # Load base model WITHOUT quantization for merging
        # Cannot merge into a quantized model
        logger.info("Loading base model without quantization for merging...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if params.BF16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load LoRA weights
        logger.info("Loading LoRA adapter weights...")
        peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)

        # Verify we have a PEFT model with merge_and_unload method
        if not hasattr(peft_model, 'merge_and_unload'):
            raise AttributeError(f"Model does not have merge_and_unload method. Got type: {type(peft_model)}")

        # Merge and unload
        logger.info("Merging LoRA weights into base model...")
        # Type ignore for Pylance - merge_and_unload returns a model, not a Tensor
        merged_model = peft_model.merge_and_unload()  # type: ignore

        # Save merged model
        logger.info(f"Saving merged model to {output_path}...")
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        logger.info(f"Merged model saved to {output_path}")


def main():
    """Command-line interface for DPO training."""
    import argparse

    parser = argparse.ArgumentParser(description="DPO training for negotiation agents")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1",
                       help="Base model name or path")
    parser.add_argument("--train-data", required=True,
                       help="Path to training DPO pairs (NDJSON)")
    parser.add_argument("--eval-data", help="Path to eval DPO pairs")
    parser.add_argument("--reference-model", help="Path to reference model")
    parser.add_argument("--output-dir", default="checkpoints/dpo",
                       help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--from-episodes", help="Generate DPO pairs from episodes file")
    parser.add_argument("--min-score-gap", type=float, default=0.3,
                       help="Minimum score gap for auto-generated pairs")

    args = parser.parse_args()

    # Initialize trainer
    trainer = DPOTrainingModule(
        model_name=args.model,
        output_dir=args.output_dir,
        reference_model_path=args.reference_model
    )

    # Train
    if args.from_episodes:
        results = trainer.train_from_episodes(
            args.from_episodes,
            min_score_gap=args.min_score_gap,
            num_train_epochs=args.epochs
        )
    else:
        results = trainer.train(
            train_data_path=args.train_data,
            eval_data_path=args.eval_data,
            num_train_epochs=args.epochs,
            resume_from_checkpoint=args.resume
        )

    print(f"\nTraining complete! Results saved to {args.output_dir}")
    print(f"Final metrics: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()