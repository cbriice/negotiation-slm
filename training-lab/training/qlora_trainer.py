"""QLoRA trainer for Mistral-7B negotiation agent training."""

import gc
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, cast
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer, DPOTrainer, SFTConfig, DPOConfig

# Import modular components
from .model_loader import ModelLoader
from .lora_config import LoRAConfigurator
from .checkpoint_manager import CheckpointManager
from .metrics_tracker import TrainingMetricsTracker
import training_parameters as params


class QLoRATrainer:
    """
    Unified QLoRA trainer for both SFT and DPO training of Mistral-7B.
    Handles model initialization, training loops, and checkpoint management.
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        output_dir: str = "models/negotiator",
        log_dir: str = "logs"
    ):
        """
        Initialize QLoRA trainer.
        
        Args:
            model_name: HuggingFace model name or local path
            output_dir: Directory for saving models
            log_dir: Directory for saving logs
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        
        # Initialize components
        self.model_loader = ModelLoader()
        self.lora_configurator = LoRAConfigurator()
        self.checkpoint_manager = CheckpointManager(base_dir=str(self.output_dir / "checkpoints"))
        
        # Model and tokenizer (to be initialized)
        self.model = None
        self.tokenizer = None
        self.metrics_tracker = None
        
        # Training state
        self.current_stage = None
        self.current_version = 1
        
    def setup_model(self, checkpoint_path: Optional[str] = None):
        """
        Setup model with 4-bit quantization and LoRA adapters.
        
        Args:
            checkpoint_path: Optional checkpoint to load from
        """
        print(f"Loading model: {self.model_name}")
        
        # Load base model with quantization
        self.model, self.tokenizer = self.model_loader.load_model(self.model_name)
        
        if checkpoint_path:
            # Load from checkpoint
            print(f"Loading checkpoint from: {checkpoint_path}")
            from peft.peft_model import PeftModel
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        else:
            # Apply LoRA configuration
            self.model = self.lora_configurator.apply_lora(self.model)
        
        # Clear cache after model setup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def train_sft(
        self,
        train_data: Union[str, Dataset],
        eval_data: Optional[Union[str, Dataset]] = None,
        num_epochs: Optional[float] = None,
        max_steps: Optional[int] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run SFT training on dialogue data.
        
        Args:
            train_data: Path to JSONL file or Dataset object
            eval_data: Optional evaluation dataset
            num_epochs: Number of epochs (overrides params if provided)
            max_steps: Maximum steps (overrides params if provided)
            resume_from_checkpoint: Resume from checkpoint
            
        Returns:
            Training metrics summary
        """
        self.current_stage = "sft"
        
        # Initialize metrics tracker
        self.metrics_tracker = TrainingMetricsTracker(
            log_dir=str(self.log_dir),
            stage=self.current_stage
        )
        
        # Setup model if not already done
        if self.model is None:
            self.setup_model(checkpoint_path=resume_from_checkpoint)
        
        # Load dataset
        print(f"Loading training data from: {train_data}")
        if isinstance(train_data, str):
            dataset = load_dataset('json', data_files=train_data)
            # Ensure we have a Dataset, not DatasetDict
            if isinstance(dataset, Dataset):
                train_dataset = dataset
            elif isinstance(dataset, DatasetDict):
                # It's a DatasetDict, get the train split or first available split
                train_dataset = dataset.get('train', list(dataset.values())[0])
            else:
                # Assume it's already a usable dataset format
                train_dataset = dataset
        else:
            train_dataset = train_data
        
        eval_dataset = None
        if eval_data:
            if isinstance(eval_data, str):
                dataset = load_dataset('json', data_files=eval_data)
                # Ensure we have a Dataset, not DatasetDict
                if isinstance(dataset, Dataset):
                    eval_dataset = dataset
                elif isinstance(dataset, DatasetDict):
                    # It's a DatasetDict, get the train split or first available split
                    eval_dataset = dataset.get('train', list(dataset.values())[0])
                else:
                    # Assume it's already a usable dataset format
                    eval_dataset = dataset
            else:
                eval_dataset = eval_data
        
        # Prepare training arguments
        training_args = self._get_sft_training_args(
            num_epochs=num_epochs,
            max_steps=max_steps or params.SFT_MAX_STEPS
        )
        
        # Initialize trainer
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call setup_model() first.")

        # Type cast datasets to ensure compatibility
        train_dataset = cast(Dataset, train_dataset)  # Type assertion for compatibility
        eval_dataset = cast(Optional[Dataset], eval_dataset) if eval_dataset else None

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            ),
            formatting_func=self._format_sft_examples,
            callbacks=[self._get_metrics_callback()]
        )
        
        # Train
        print(f"Starting SFT training for {training_args.num_train_epochs} epochs")
        start_time = time.time()
        
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint if resume_from_checkpoint else None)
        
        # Save checkpoint
        final_metrics = {
            "train_loss": train_result.metrics.get("train_loss", 0),
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0)
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            tokenizer=self.tokenizer,
            stage=self.current_stage,
            version=self.current_version,
            metrics=final_metrics,
            training_args=training_args.to_dict()
        )
        
        # Save training summary
        summary = self.metrics_tracker.save_training_summary()
        
        # Cleanup
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f"SFT training completed in {time.time() - start_time:.1f}s")
        print(f"Model saved to: {checkpoint_path}")
        
        self.current_version += 1
        return summary
    
    def train_dpo(
        self,
        train_data: Union[str, Dataset],
        reference_model_path: str,
        eval_data: Optional[Union[str, Dataset]] = None,
        num_epochs: Optional[float] = None,
        max_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run DPO training on preference data.
        
        Args:
            train_data: Path to DPO pairs JSONL or Dataset
            reference_model_path: Path to reference model checkpoint
            eval_data: Optional evaluation dataset
            num_epochs: Number of epochs (overrides params if provided)
            max_steps: Maximum steps (overrides params if provided)
            
        Returns:
            Training metrics summary
        """
        self.current_stage = "dpo"
        
        # Initialize metrics tracker
        self.metrics_tracker = TrainingMetricsTracker(
            log_dir=str(self.log_dir),
            stage=self.current_stage
        )
        
        # Setup model if not already done
        if self.model is None:
            self.setup_model()
        
        # Load reference model
        print(f"Loading reference model from: {reference_model_path}")
        ref_model = self.model_loader.load_reference_model(reference_model_path)
        
        # Load dataset
        print(f"Loading DPO training data from: {train_data}")
        if isinstance(train_data, str):
            train_dataset = load_dataset('json', data_files=train_data, split='train')
        else:
            train_dataset = train_data
        
        eval_dataset = None
        if eval_data:
            if isinstance(eval_data, str):
                dataset = load_dataset('json', data_files=eval_data)
                # Ensure we have a Dataset, not DatasetDict
                if isinstance(dataset, Dataset):
                    eval_dataset = dataset
                elif isinstance(dataset, DatasetDict):
                    # It's a DatasetDict, get the train split or first available split
                    eval_dataset = dataset.get('train', list(dataset.values())[0])
                else:
                    # Assume it's already a usable dataset format
                    eval_dataset = dataset
            else:
                eval_dataset = eval_data
        
        # Prepare training arguments
        training_args = self._get_dpo_training_args(
            num_epochs=num_epochs,
            max_steps=max_steps or params.DPO_MAX_STEPS
        )
        
        # Initialize DPO trainer
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call setup_model() first.")

        # Type cast datasets to ensure compatibility
        train_dataset = cast(Dataset, train_dataset)  # Type assertion for compatibility
        eval_dataset = cast(Optional[Dataset], eval_dataset) if eval_dataset else None

        trainer = DPOTrainer(
            model=self.model,
            ref_model=ref_model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            callbacks=[self._get_dpo_metrics_callback()]
        )
        
        # Train
        print(f"Starting DPO training for {training_args.num_train_epochs} epochs")
        start_time = time.time()
        
        train_result = trainer.train()
        
        # Save checkpoint
        final_metrics = {
            "train_loss": train_result.metrics.get("train_loss", 0),
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0)
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            tokenizer=self.tokenizer,
            stage=self.current_stage,
            version=self.current_version,
            metrics=final_metrics,
            training_args=training_args.to_dict()
        )
        
        # Save training summary
        summary = self.metrics_tracker.save_training_summary()
        
        # Cleanup
        del trainer
        del ref_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f"DPO training completed in {time.time() - start_time:.1f}s")
        print(f"Model saved to: {checkpoint_path}")
        
        self.current_version += 1
        return summary
    
    def _get_sft_training_args(self, num_epochs: Optional[float], max_steps: int) -> SFTConfig:
        """Get SFT training arguments from parameters."""
        return SFTConfig(
            output_dir=str(self.output_dir / "sft_checkpoints"),
            num_train_epochs=num_epochs if num_epochs is not None else (max_steps / 1000),  # Rough estimate
            max_steps=max_steps,
            per_device_train_batch_size=params.PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=params.GRAD_ACCUM_STEPS,
            learning_rate=params.SFT_LR,
            warmup_ratio=params.SFT_WARMUP_RATIO,
            lr_scheduler_type=params.SFT_LR_SCHEDULER,
            optim=params.OPTIM,
            bf16=params.BF16,
            gradient_checkpointing=params.GRADIENT_CHECKPOINTING,
            logging_steps=params.LOGGING_STEPS,
            save_steps=params.SAVE_STEPS,
            save_strategy=params.SAVE_STRATEGY,
            evaluation_strategy=params.EVAL_STRATEGY,
            seed=params.SEED,
            remove_unused_columns=False,
            report_to="none",  # Disable wandb/tensorboard by default
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            max_seq_length=params.MAX_SEQ_LEN,
            packing=params.PACKING
        )
    
    def _get_dpo_training_args(self, num_epochs: Optional[float], max_steps: int) -> DPOConfig:
        """Get DPO training arguments from parameters."""
        return DPOConfig(
            output_dir=str(self.output_dir / "dpo_checkpoints"),
            beta=params.DPO_BETA,
            label_smoothing=params.DPO_LABEL_SMOOTH,
            max_prompt_length=params.DPO_MAX_PROMPT_LEN,
            max_length=params.MAX_SEQ_LEN,
            num_train_epochs=num_epochs if num_epochs is not None else (max_steps / 1000),
            max_steps=max_steps,
            per_device_train_batch_size=params.PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=params.GRAD_ACCUM_STEPS,
            learning_rate=params.DPO_LR,
            warmup_ratio=params.SFT_WARMUP_RATIO,  # Use same warmup
            lr_scheduler_type=params.SFT_LR_SCHEDULER,  # Use same scheduler
            optim=params.OPTIM,
            bf16=params.BF16,
            gradient_checkpointing=params.GRADIENT_CHECKPOINTING,
            logging_steps=params.LOGGING_STEPS,
            save_steps=params.SAVE_STEPS,
            save_strategy=params.SAVE_STRATEGY,
            evaluation_strategy=params.EVAL_STRATEGY,
            seed=params.SEED,
            remove_unused_columns=False,
            report_to="none",
            dataloader_num_workers=4,
            dataloader_pin_memory=True
        )
    
    def _format_sft_examples(self, examples: Dict[str, Any]) -> List[str]:
        """Format examples for SFT training."""
        formatted = []
        
        for i in range(len(examples['dialogue'])):
            # Format dialogue into chat format
            dialogue = examples['dialogue'][i]
            text = ""
            
            for turn in dialogue:
                role = turn.get('role', 'user')
                content = turn.get('content', turn.get('text', ''))
                text += f"{role}: {content}\n"
            
            formatted.append(text)
        
        return formatted
    
    def _get_metrics_callback(self):
        """Get callback for logging metrics during SFT training."""
        from transformers import TrainerCallback
        
        class MetricsCallback(TrainerCallback):
            def __init__(self, metrics_tracker):
                self.metrics_tracker = metrics_tracker
            
            def on_log(self, args, state, control, logs=None, **kwargs):  # pylint: disable=unused-argument
                if logs:
                    # Extract relevant metrics
                    metrics = {
                        "train_loss": logs.get("loss"),
                        "learning_rate": logs.get("learning_rate"),
                        "gradient_norm": logs.get("grad_norm"),
                    }
                    
                    # Calculate tokens per second if available
                    if "train_runtime" in logs and "total_flos" in logs:
                        runtime = logs["train_runtime"]
                        if runtime > 0:
                            # Estimate tokens from FLOPS (rough approximation)
                            metrics["tokens_per_second"] = logs["total_flos"] / (runtime * 1e12)
                    
                    # Log to tracker
                    self.metrics_tracker.log_training_step(
                        step=state.global_step,
                        epoch=state.epoch or 0,
                        metrics={k: v for k, v in metrics.items() if v is not None}
                    )
        
        return MetricsCallback(self.metrics_tracker)
    
    def _get_dpo_metrics_callback(self):
        """Get callback for logging metrics during DPO training."""
        from transformers import TrainerCallback
        
        class DPOMetricsCallback(TrainerCallback):
            def __init__(self, metrics_tracker):
                self.metrics_tracker = metrics_tracker
            
            def on_log(self, args, state, control, logs=None, **kwargs):  # pylint: disable=unused-argument
                if logs:
                    # Extract DPO-specific metrics
                    metrics = {
                        "train_loss": logs.get("loss"),
                        "learning_rate": logs.get("learning_rate"),
                        "gradient_norm": logs.get("grad_norm"),
                        "reward_margin": logs.get("rewards/margins", logs.get("reward_margin")),
                        "reward_chosen": logs.get("rewards/chosen", logs.get("reward_chosen")),
                        "reward_rejected": logs.get("rewards/rejected", logs.get("reward_rejected")),
                        "kl_divergence": logs.get("kl", logs.get("kl_div"))
                    }
                    
                    # Log to tracker
                    self.metrics_tracker.log_training_step(
                        step=state.global_step,
                        epoch=state.epoch or 0,
                        metrics={k: v for k, v in metrics.items() if v is not None}
                    )
        
        return DPOMetricsCallback(self.metrics_tracker)
    
    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: Optional[float] = None,
        top_p: float = 0.95,
        do_sample: bool = True
    ) -> str:
        """
        Generate response from the model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature (defaults to params.MODEL_TEMP)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        temperature = temperature if temperature is not None else params.MODEL_TEMP
        
        # Tokenize input
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call setup_model() first.")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else None,
                eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else None
            )
        
        # Decode
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized.")
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def load_checkpoint(self, checkpoint_name: str):
        """
        Load a saved checkpoint.
        
        Args:
            checkpoint_name: Name or path of checkpoint to load
        """
        checkpoint_info = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        checkpoint_path = checkpoint_info["checkpoint_path"]
        
        print(f"Loading checkpoint: {checkpoint_name}")
        print(f"Stage: {checkpoint_info.get('stage')}, Version: {checkpoint_info.get('version')}")
        
        # Setup model with checkpoint
        self.setup_model(checkpoint_path=checkpoint_path)
        
        # Update state
        self.current_stage = checkpoint_info.get('stage')
        self.current_version = checkpoint_info.get('version', 1) + 1
        
        return checkpoint_info