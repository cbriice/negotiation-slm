#!/usr/bin/env python3
"""Test script to verify TensorBoard integration."""

import torch
import numpy as np
from pathlib import Path
import time

# Import our TensorBoard writer
from utils.tensorboard_writer import TensorBoardWriter

# Import error handlers for testing
from utils.batch_size_optimizer import BatchSizeOptimizer, BatchSizeConfig
from utils.gradient_monitor import GradientMonitor, GradientHealthConfig
from utils.loss_validator import LossValidator, LossValidationConfig
from utils.precision_handler import PrecisionHandler, PrecisionConfig


def test_basic_logging():
    """Test basic TensorBoard logging functionality."""
    print("Testing basic TensorBoard logging...")

    # Initialize writer
    tb_writer = TensorBoardWriter(
        log_dir="runs/test",
        experiment_name="test_run",
        comment="integration_test"
    )

    # Simulate training loop
    for step in range(100):
        # Log loss
        loss = 2.0 * np.exp(-step / 20) + 0.1 * np.random.random()
        tb_writer.log_loss_metrics(loss, step, loss_type="train")

        # Log eval loss every 10 steps
        if step % 10 == 0:
            eval_loss = loss + 0.05 * np.random.random()
            tb_writer.log_loss_metrics(eval_loss, step, loss_type="eval")

        # Log learning rate
        lr = 1e-3 * (0.95 ** (step // 20))
        tb_writer.log_learning_rate(lr, step)

        # Log performance metrics
        tb_writer.log_performance_metrics(
            tokens_per_second=1000 + 100 * np.random.random(),
            samples_per_second=32 + 4 * np.random.random(),
            step=step
        )

        # Log DPO metrics (simulate)
        if step % 5 == 0:
            chosen_reward = 0.5 + 0.1 * np.random.random()
            rejected_reward = 0.2 + 0.1 * np.random.random()
            tb_writer.log_dpo_metrics(
                chosen_rewards=chosen_reward,
                rejected_rewards=rejected_reward,
                reward_margin=chosen_reward - rejected_reward,
                step=step
            )

    # Log final hyperparameters
    hparams = {
        "learning_rate": 1e-3,
        "batch_size": 16,
        "model": "test_model",
        "optimizer": "adamw"
    }
    metrics = {
        "final_loss": 0.15,
        "total_steps": 100,
        "best_accuracy": 0.95
    }
    tb_writer.log_hyperparameters(hparams, metrics)

    tb_writer.close()
    print(f"✓ Basic logging test completed. Check TensorBoard at: {tb_writer.log_dir}")


def test_error_handler_logging():
    """Test error handler integration."""
    print("\nTesting error handler logging...")

    # Initialize writer
    tb_writer = TensorBoardWriter(
        log_dir="runs/test",
        experiment_name="error_handler_test",
        comment="error_handling"
    )

    # Create a dummy model for gradient monitoring
    dummy_model = torch.nn.Linear(10, 10)

    # Initialize error handlers
    batch_optimizer = BatchSizeOptimizer(
        config=BatchSizeConfig(
            initial_batch_size=16,
            min_batch_size=1,
            max_batch_size=32
        )
    )

    gradient_monitor = GradientMonitor(
        config=GradientHealthConfig(
            max_gradient_norm=10.0,
            explosion_threshold=100.0
        ),
        model=dummy_model  # Pass the dummy model
    )

    loss_validator = LossValidator(
        config=LossValidationConfig(
            max_loss=10.0,
            min_loss=0.0
        )
    )

    precision_handler = PrecisionHandler(
        config=PrecisionConfig(
            use_bf16=True,
            use_fp16=False
        )
    )

    # Simulate training with error conditions
    for step in range(50):
        # Generate new gradients for each step
        dummy_input = torch.randn(1, 10)
        dummy_output = dummy_model(dummy_input)
        model_loss = dummy_output.sum()
        dummy_model.zero_grad()
        model_loss.backward()

        # Simulate loss values for validation
        if step == 20:
            # Simulate loss spike
            loss = torch.tensor(15.0)
        elif step == 30:
            # Simulate NaN
            loss = torch.tensor(float('nan'))
        else:
            loss = torch.tensor(2.0 * np.exp(-step / 10) + 0.1 * np.random.random())

        # Validate loss
        is_valid, action, info = loss_validator.validate_loss(loss, step)

        if not is_valid:
            tb_writer.log_error_event(
                error_type="loss_validation",
                error_message=f"Action: {action}, Info: {info}",
                step=step,
                severity="warning" if action == "skip_update" else "error"
            )

        # Log all error handler metrics
        tb_writer.log_all_error_handlers(
            batch_optimizer=batch_optimizer,
            gradient_monitor=gradient_monitor,
            loss_validator=loss_validator,
            precision_handler=precision_handler,
            step=step
        )

        # Simulate batch size adjustment
        if step == 25:
            batch_optimizer.reduce_batch_size_on_oom()
            tb_writer.log_error_event(
                error_type="OOM",
                error_message="Reduced batch size due to OOM",
                step=step,
                severity="warning"
            )

    tb_writer.close()
    print(f"✓ Error handler test completed. Check TensorBoard at: {tb_writer.log_dir}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TensorBoard Integration Test")
    print("=" * 60)

    # Create test directory
    Path("runs/test").mkdir(parents=True, exist_ok=True)

    # Run tests
    test_basic_logging()
    test_error_handler_logging()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("To view results, run: tensorboard --logdir=runs")
    print("=" * 60)


if __name__ == "__main__":
    main()