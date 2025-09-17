
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

