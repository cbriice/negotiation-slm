1. QLoRA Trainer Module (training/qlora_trainer.py)

  Core components:
  - SFTTrainer wrapper for negotiation dialogue fine-tuning
  - DPOTrainer wrapper for preference optimization
  - Model loading with 4-bit quantization via bitsandbytes
  - LoRA adapter configuration from training_parameters.py
  - Checkpoint management with versioning

Detailed Implementation Plan: QLoRA Trainer Module

  File: training/qlora_trainer.py

  Core Architecture

  class QLoRATrainer:
      """
      Unified QLoRA trainer for both SFT and DPO training of Mistral-7B.
      Handles model initialization, training loops, and checkpoint management.
      """

      Components:
      ├── Model Loading & Quantization
      ├── LoRA Configuration
      ├── Training Orchestration
      ├── Checkpoint Management
      └── Memory Optimization

  1. Model Initialization (__init__ and setup_model)

  Features:
  - Load Mistral-7B with 4-bit quantization using bitsandbytes
  - Configure BitsAndBytesConfig with nf4 quantization
  - Initialize LoRA adapters via PEFT
  - Setup tokenizer with proper padding tokens
  - Device mapping for RTX 5070 Ti optimization

  Key Parameters (from training_parameters.py):
  - LOAD_IN_4BIT = True
  - BNB_4BIT_QUANT_TYPE = "nf4"
  - BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
  - R = 16, ALPHA = 32, DROPOUT = 0.05
  - TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  Other parameters:
  - TRL/Trainer: packing = True

  **Important**: ANY and ALL parameters used in training code should be housed in training_parameters.py. if they don't already exist, initialize them in training_parameters.py and NO default, hardcoded parameters should exist

  2. SFT Training Pipeline (train_sft method)

  Workflow:
  1. Load JSONL dataset via datasets.load_dataset()
  2. Apply chat template formatting
  3. Initialize SFTTrainer with:
     - model & tokenizer
     - TrainingArguments from parameters
     - DataCollatorForLanguageModeling
  4. Execute training with gradient accumulation
  5. Save LoRA adapters periodically

  Special Handling:
  - Handles two types of responses/training data formatting (responses + strategy)
  - Dynamic sequence length based on batch
  - Automatic mixed precision (bf16)
  - Gradient checkpointing for memory efficiency

  3. DPO Training Pipeline (train_dpo method)

  Workflow:
  1. Load DPO pairs from JSONL
  2. Format as preference dataset
  3. Initialize DPOTrainer with:
     - model & reference model
     - DPO beta = 0.1
     - Label smoothing = 0.0
  4. Track reward margins during training
  5. Early stopping on reward divergence

  Key Features:
  - Reference model frozen in 4-bit
  - Chosen/rejected response processing
  - KL divergence monitoring
  - Reward difference tracking

  4. Checkpoint Management (CheckpointManager inner class)

  Features:
  - Versioning: negotiation_{stage}_{version}_{timestamp}
  - Metadata tracking in checkpoint_info.json:
    - Training parameters
    - Dataset statistics
    - Performance metrics
    - Git commit hash
  - Adapter-only saves (small files ~30MB)
  - Automatic best model tracking
  - Rollback capability

  Directory Structure:
  checkpoints/
  ├── negotiation_baseline_v1_20250113/
  │   ├── adapter_model.bin
  │   ├── adapter_config.json
  │   └── checkpoint_info.json
  ├── negotiation_sft_v2_20250114/
  └── negotiation_dpo_v3_20250115/

  ===================================================== ABOVE HERE IS DONE ===================================================

  =========================================================== TODO ===========================================================

  6. Training Monitoring (MetricsTracker inner class)

  Tracked Metrics:
  - Loss curves (train/eval)
  - Learning rate schedule
  - Gradient norms
  - Token throughput (tps)
  - VRAM usage
  - Reward margins (DPO only)

  Logging:
  - TensorBoard integration
  - CSV export for analysis
  - Real-time console output
  - Checkpoint performance summary

  7. Integration Points

  Data Pipeline:
  - Reads from EpisodeLogger JSONL outputs
  - Uses data_converter.py transformations
  - Validates with protocol_validator.py

  Configuration:
  - All hyperparameters from training_parameters.py
  - No hardcoded values
  - Runtime validation of parameters

  Testing:
  - Integration with existing test suite
  - Mock training for CI/CD
  - Memory profiling utilities

  8. Usage API

  # Initialize trainer
  trainer = QLoRATrainer(
      model_name="mistralai/Mistral-7B-v0.1",
      output_dir="models/negotiator"
  )

  # SFT training
  trainer.train_sft(
      train_data="data/training/sft/high_quality.jsonl",
      eval_data="data/training/sft/eval.jsonl",
      num_epochs=0.5
  )

  # DPO training
  trainer.train_dpo(
      train_data="data/training/dpo/pairs.jsonl",
      reference_model_path="models/negotiator/sft_best",
      num_epochs=1
  )

  # Load checkpoint
  trainer.load_checkpoint("negotiation_sft_v2_20250114")

  # Inference mode
  response = trainer.generate(
      prompt="I'd like to buy this item for $50",
      max_length=256,
      temperature=0.5
  )

  9. Error Handling & Recovery

  Safeguards:
  - Automatic checkpoint saving on crash
  - Training state persistence
  - OOM recovery with batch size reduction
  - Gradient explosion detection
  - NaN loss handling
  - Automatic mixed precision fallback

  10. Performance Optimizations

  Specific to RTX 5070 Ti:
  - Flash Attention 2 if available
  - Tensor cores via bf16
  - Optimized data loading (num_workers=4)
  - Pinned memory transfers
  - Efficient tokenization caching

  Complementary Files

  1. training/trainer_utils.py - Helper functions for data loading, formatting
  2. training/callbacks.py - Custom callbacks for monitoring, early stopping
  3. training/model_registry.py - Track all trained models and versions 

### END part 1
--------------------------------------------------------------------------------------------------------------------------------------

Revised Data Annotation System

  What Already Exists:

  - Protocol validation & scoring - protocol_validator.py handles violations, scores, tag leak detection
  - Episode metrics - episode_metrics.py calculates deal outcomes, token counts, negotiation dynamics
  - Quality filtering - episode_logger.py has filter_by_quality(), validate_episode_for_training()
  - DPO pair selection - episode_logger.py has select_and_log_dpo_pairs(), batch_create_dpo_pairs()

  What's Actually Needed:

  1. Annotation Interface (training/annotation/labeler.py)

  Components:
  - Interactive CLI for human review of edge cases
  - Batch annotation with customizable criteria
  - Integration with existing scoring functions
  - Export annotations as metadata overlay

  2. Advanced Pair Selection (training/annotation/pair_selector.py)

  Features:
  - Curriculum-based selection (easy → hard pairs)
  - Diversity sampling to avoid overfitting
  - Turn-specific pair generation (not just episode-level)
  - Multi-criteria optimization beyond just protocol score

  3. Quality Assurance (training/annotation/qa.py)

  Purpose:
  - Detect data drift between training stages
  - Identify outliers that existing metrics miss
  - Validate DPO pair contrast strength
  - Generate training readiness reports

  Integration Points:

  - Use existing calculate_protocol_score() and EpisodeMetrics.calculate()
  - Extend select_dpo_pairs() with advanced selection strategies
  - Leverage validate_episode_for_training() as base validator
  - Build on transform_to_sft_packet() and transform_to_dpo_pair()

  The annotation system becomes a thin orchestration layer that coordinates existing functionality rather than reimplementing it. The focus shifts to human-in-the-loop
  review, advanced selection algorithms, and quality assurance - things not already covered.

-------------------------------------------------------------------------------------------------------------------------------------------

4. Model Versioning System (training/model_manager.py)

  Features:
  - Checkpoint naming convention: negotiation_{phase}_{version}_{timestamp}
  - Metadata tracking (training params, dataset stats)
  - Model registry with performance metrics
  - Rollback capability for regression issues

5. Training Pipeline Orchestrator (training/pipeline.py)

  Workflow:
  1. Load episodes from NDJSON logs
  2. Run annotation pipeline
  3. Generate training datasets (SFT/DPO)
  4. Initialize QLoRA trainer
  5. Execute training with monitoring
  6. Save checkpoints and metrics
  7. Validate against held-out test set

