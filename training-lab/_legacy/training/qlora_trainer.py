"""
QLoRA trainer wrapper for Mistral-7B negotiation agent training.
This module acts as a unified interface that delegates to specialized trainers (SFT, DPO)
while managing common components like model loading and checkpoint management.
"""

import gc
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
from datasets import Dataset

# Import specialized trainers
from .sft_trainer import SFTTrainingPipeline
from .dpo_trainer import DPOTrainingModule

# Import common components
from .model_loader import ModelLoader
from .lora_config import LoRAConfigurator
from .checkpoint_manager import CheckpointManager
from .metrics_tracker import TrainingMetricsTracker

# Import unified dataset loader
from utils.dataset_loader import UnifiedDatasetLoader
from utils.tensorboard_writer import TensorBoardWriter

import training_parameters as params


class QLoRATrainer:
    """
    Unified QLoRA trainer that delegates to specialized SFT and DPO trainers.
    Provides a consistent interface for ParlAI integration and manages shared resources.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        output_dir: str = "models/negotiator",
        log_dir: str = "logs"
    ):
        """
        Initialize QLoRA trainer wrapper.

        Args:
            model_name: HuggingFace model name or local path
            output_dir: Directory for saving models
            log_dir: Directory for saving logs
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize common components
        self.model_loader = ModelLoader()
        self.lora_configurator = LoRAConfigurator()
        self.checkpoint_manager = CheckpointManager(
            base_dir=str(self.output_dir / "checkpoints")
        )
        self.dataset_loader = UnifiedDatasetLoader()

        # Initialize specialized trainers (lazy initialization)
        self._sft_trainer = None
        self._dpo_trainer = None

        # Current model and tokenizer (shared across trainers)
        self.model = None
        self.tokenizer = None

        # Training state
        self.current_stage = None
        self.current_version = 1

        # TensorBoard writer (shared across trainers)
        self.tb_writer = None

    @property
    def sft_trainer(self) -> SFTTrainingPipeline:
        """Get or create SFT trainer instance."""
        if self._sft_trainer is None:
            self._sft_trainer = SFTTrainingPipeline(
                model_name=self.model_name,
                output_dir=str(self.output_dir / "sft"),
                checkpoint_dir=str(self.output_dir / "checkpoints" / "sft")
            )
            # Share model and tokenizer if already loaded
            if self.model is not None:
                self._sft_trainer.model = self.model
                self._sft_trainer.tokenizer = self.tokenizer
        return self._sft_trainer

    @property
    def dpo_trainer(self) -> DPOTrainingModule:
        """Get or create DPO trainer instance."""
        if self._dpo_trainer is None:
            self._dpo_trainer = DPOTrainingModule(
                model_name=self.model_name,
                output_dir=str(self.output_dir / "dpo"),
                reference_model_path=None  # Will be set when needed
            )
        return self._dpo_trainer

    def setup_model(self, checkpoint_path: Optional[str] = None):
        """
        Setup model with 4-bit quantization and LoRA adapters.
        This is the centralized model loading that can be shared.

        Args:
            checkpoint_path: Optional checkpoint to load from
        """
        print(f"Loading model: {self.model_name}")

        # Load base model with quantization
        self.model, self.tokenizer = self.model_loader.load_model(self.model_name)

        if checkpoint_path:
            # Load from checkpoint
            print(f"Loading checkpoint from: {checkpoint_path}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        else:
            # Apply LoRA configuration
            self.model = self.lora_configurator.apply_lora(self.model)

        # Share with specialized trainers if they exist
        if self._sft_trainer is not None:
            self._sft_trainer.model = self.model
            self._sft_trainer.tokenizer = self.tokenizer

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
        resume_from_checkpoint: Optional[str] = None,
        use_specialized_trainer: bool = True
    ) -> Dict[str, Any]:
        """
        Run SFT training on dialogue data.

        Args:
            train_data: Path to JSONL file or Dataset object
            eval_data: Optional evaluation dataset
            num_epochs: Number of epochs (overrides params if provided)
            max_steps: Maximum steps (overrides params if provided)
            resume_from_checkpoint: Resume from checkpoint
            use_specialized_trainer: If True, delegate to SFTTrainingPipeline

        Returns:
            Training metrics summary
        """
        self.current_stage = "sft"

        if use_specialized_trainer:
            # Delegate to specialized SFT trainer
            print("Delegating to specialized SFT trainer...")

            # Load datasets using unified loader
            if isinstance(train_data, str):
                train_dataset = self.dataset_loader.load_sft_dataset(train_data)
            else:
                train_dataset = train_data

            eval_dataset = None
            if eval_data:
                if isinstance(eval_data, str):
                    eval_dataset = self.dataset_loader.load_sft_dataset(eval_data)
                else:
                    eval_dataset = eval_data

            # Setup model in specialized trainer if needed
            if self.sft_trainer.model is None:
                if self.model is None:
                    self.setup_model(checkpoint_path=resume_from_checkpoint)
                self.sft_trainer.model = self.model
                self.sft_trainer.tokenizer = self.tokenizer

            # Initialize TensorBoard if not already done
            if self.tb_writer is None:
                self.tb_writer = TensorBoardWriter(
                    log_dir=str(self.output_dir / "runs"),
                    experiment_name=f"qlora_v{self.current_version}",
                    comment=self.current_stage or "training"
                )
                # Share with SFT trainer
                self.sft_trainer.tb_writer = self.tb_writer

            # Run training
            metrics = self.sft_trainer.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                num_epochs=num_epochs,
                max_steps=max_steps,
                output_suffix=f"_v{self.current_version}"
            )

            # Update our model reference
            self.model = self.sft_trainer.model
            self.tokenizer = self.sft_trainer.tokenizer

        else:
            # Use lightweight built-in training (minimal implementation)
            # This path is kept for backward compatibility but delegates most work
            raise NotImplementedError(
                "Direct SFT training not implemented. Use specialized trainer."
            )

        self.current_version += 1
        return metrics

    def train_dpo(
        self,
        train_data: Union[str, Dataset],
        reference_model_path: str,
        eval_data: Optional[Union[str, Dataset]] = None,
        num_epochs: Optional[float] = None,
        max_steps: Optional[int] = None,
        use_specialized_trainer: bool = True
    ) -> Dict[str, Any]:
        """
        Run DPO training on preference data.

        Args:
            train_data: Path to DPO pairs JSONL or Dataset
            reference_model_path: Path to reference model checkpoint
            eval_data: Optional evaluation dataset
            num_epochs: Number of epochs (overrides params if provided)
            max_steps: Maximum steps (overrides params if provided)
            use_specialized_trainer: If True, delegate to DPOTrainingModule

        Returns:
            Training metrics summary
        """
        self.current_stage = "dpo"

        if use_specialized_trainer:
            # Delegate to specialized DPO trainer
            print("Delegating to specialized DPO trainer...")

            # Update reference model path
            self.dpo_trainer.reference_model_path = reference_model_path

            # Share TensorBoard writer if available
            if self.tb_writer:
                self.dpo_trainer.tb_writer = self.tb_writer
            else:
                # Initialize if needed
                self.tb_writer = TensorBoardWriter(
                    log_dir=str(self.output_dir / "runs"),
                    experiment_name=f"qlora_v{self.current_version}",
                    comment="dpo"
                )
                self.dpo_trainer.tb_writer = self.tb_writer

            # Convert string paths to actual paths for the trainer
            if isinstance(train_data, str):
                train_path = train_data
            else:
                # Save dataset to temp file if it's a Dataset object
                temp_path = self.output_dir / "temp_dpo_train.jsonl"
                train_data.to_json(str(temp_path), lines=True)
                train_path = str(temp_path)

            eval_path = None
            if eval_data:
                if isinstance(eval_data, str):
                    eval_path = eval_data
                else:
                    temp_eval_path = self.output_dir / "temp_dpo_eval.jsonl"
                    eval_data.to_json(str(temp_eval_path), lines=True)
                    eval_path = str(temp_eval_path)

            # Run training
            metrics = self.dpo_trainer.train(
                train_data_path=train_path,
                eval_data_path=eval_path,
                num_train_epochs=num_epochs if num_epochs else None,
                resume_from_checkpoint=None
            )

        else:
            # Use lightweight built-in training
            raise NotImplementedError(
                "Direct DPO training not implemented. Use specialized trainer."
            )

        self.current_version += 1
        return metrics

    def train_from_episodes(
        self,
        episodes_path: str,
        training_type: str = "dpo",
        min_score_gap: float = 0.3,
        num_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train directly from episode files.

        Args:
            episodes_path: Path to episodes NDJSON file
            training_type: "dpo" or "sft"
            min_score_gap: For DPO, minimum score gap for pair selection
            num_epochs: Number of training epochs

        Returns:
            Training results
        """
        if training_type == "dpo":
            # Use DPO trainer's built-in episode handling
            return self.dpo_trainer.train_from_episodes(
                episodes_path=episodes_path,
                min_score_gap=min_score_gap,
                num_train_epochs=num_epochs
            )
        elif training_type == "sft":
            # Convert episodes to SFT format
            from training.data_converter import DataConverter

            converter = DataConverter(output_dir=str(self.output_dir / "data"))
            sft_file = converter.episodes_to_sft(
                episodes_path,
                output_file=str(self.output_dir / "data" / "episodes_sft.jsonl")
            )

            # Train with SFT
            return self.train_sft(
                train_data=sft_file,
                num_epochs=num_epochs
            )
        else:
            raise ValueError(f"Unknown training type: {training_type}")

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
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
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

    def merge_lora_weights(self, checkpoint_path: str, output_path: str):
        """
        Merge LoRA weights with base model and save.
        Delegates to DPO trainer's implementation.

        Args:
            checkpoint_path: Path to LoRA checkpoint
            output_path: Path to save merged model
        """
        self.dpo_trainer.merge_and_save(checkpoint_path, output_path)

    def cleanup(self):
        """Clean up resources and free memory."""
        # Clean up model
        if self.model is not None:
            del self.model
            self.model = None

        # Clean up tokenizer
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clean up specialized trainers
        if self._sft_trainer is not None:
            self._sft_trainer.model = None
            self._sft_trainer.tokenizer = None
            del self._sft_trainer
            self._sft_trainer = None

        if self._dpo_trainer is not None:
            del self._dpo_trainer
            self._dpo_trainer = None

        # Close TensorBoard writer
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None

        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print("Resources cleaned up successfully")