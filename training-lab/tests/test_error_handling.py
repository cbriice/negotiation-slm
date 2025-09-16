"""
Test script for validating error handling functionality in DPO and SFT trainers.
Simulates various error conditions to ensure proper recovery mechanisms.
"""

import torch
import numpy as np
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import gc

# Configure logging to see all error handler messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import error handling modules
from utils.error_handler import ErrorHandler
from utils.training_state import TrainingStateManager
from utils.batch_size_optimizer import BatchSizeOptimizer, BatchSizeConfig
from utils.gradient_monitor import GradientMonitor, GradientHealthConfig
from utils.loss_validator import LossValidator, LossValidationConfig
from utils.precision_handler import PrecisionHandler, PrecisionConfig


class ErrorSimulator:
    """Simulates various error conditions for testing."""

    def __init__(self):
        self.test_results = {}
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Created temp directory: {self.temp_dir}")

    def cleanup(self):
        """Clean up temp directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temp directory")

    def create_mock_model(self, params: int = 1000000):
        """Create a mock model for testing."""
        model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )

        # Initialize with normal parameters
        for param in model.parameters():
            param.data.normal_(0, 0.02)

        return model

    def test_oom_recovery(self):
        """Test OOM error recovery with batch size reduction."""
        logger.info("\n" + "="*50)
        logger.info("Testing OOM Recovery...")
        logger.info("="*50)

        try:
            # Initialize batch size optimizer
            batch_optimizer = BatchSizeOptimizer(
                config=BatchSizeConfig(
                    initial_batch_size=32,
                    min_batch_size=1,
                    max_batch_size=64,
                    gradient_accumulation_steps=1,
                    reduction_factor=0.5
                ),
                log_dir=str(self.temp_dir)
            )

            # Simulate OOM by adding current batch size to OOM set
            batch_optimizer.oom_batch_sizes.add(32)

            # Test batch size reduction
            new_batch_size, new_grad_accum = batch_optimizer.reduce_batch_size_on_oom()

            assert new_batch_size == 16, f"Expected batch size 16, got {new_batch_size}"
            logger.info(f"✓ Batch size reduced from 32 to {new_batch_size}")

            # Test further reduction
            batch_optimizer.current_batch_size = new_batch_size
            batch_optimizer.oom_batch_sizes.add(16)
            new_batch_size, new_grad_accum = batch_optimizer.reduce_batch_size_on_oom()

            assert new_batch_size == 8, f"Expected batch size 8, got {new_batch_size}"
            logger.info(f"✓ Batch size further reduced to {new_batch_size}")

            # Test minimum batch size with gradient accumulation increase
            batch_optimizer.current_batch_size = 1
            new_batch_size, new_grad_accum = batch_optimizer.reduce_batch_size_on_oom()

            assert new_batch_size == 1, "Batch size should stay at minimum"
            assert new_grad_accum > 1, "Gradient accumulation should increase"
            logger.info(f"✓ At minimum batch size, increased grad accumulation to {new_grad_accum}")

            self.test_results["oom_recovery"] = "PASSED"

        except Exception as e:
            logger.error(f"OOM recovery test failed: {e}")
            self.test_results["oom_recovery"] = f"FAILED: {e}"

    def test_nan_loss_handling(self):
        """Test NaN loss detection and handling."""
        logger.info("\n" + "="*50)
        logger.info("Testing NaN Loss Handling...")
        logger.info("="*50)

        try:
            # Initialize loss validator
            loss_validator = LossValidator(
                config=LossValidationConfig(
                    nan_action="skip",
                    inf_action="skip"
                )
            )

            # Test NaN loss
            nan_loss = torch.tensor(float('nan'))
            is_valid, action, info = loss_validator.validate_loss(nan_loss, step=1)

            assert not is_valid, "NaN loss should be invalid"
            assert action == "skip_update", f"Expected 'skip_update', got {action}"
            logger.info("✓ NaN loss detected and marked for skipping")

            # Test Inf loss
            inf_loss = torch.tensor(float('inf'))
            is_valid, action, info = loss_validator.validate_loss(inf_loss, step=2)

            assert not is_valid, "Inf loss should be invalid"
            assert action == "skip_update", f"Expected 'skip_update', got {action}"
            logger.info("✓ Inf loss detected and marked for skipping")

            # Test normal loss
            normal_loss = torch.tensor(0.5)
            is_valid, action, info = loss_validator.validate_loss(normal_loss, step=3)

            assert is_valid, "Normal loss should be valid"
            assert action == "continue", f"Expected 'continue', got {action}"
            logger.info("✓ Normal loss validated successfully")

            self.test_results["nan_loss_handling"] = "PASSED"

        except Exception as e:
            logger.error(f"NaN loss handling test failed: {e}")
            self.test_results["nan_loss_handling"] = f"FAILED: {e}"

    def test_gradient_explosion(self):
        """Test gradient explosion detection and clipping."""
        logger.info("\n" + "="*50)
        logger.info("Testing Gradient Explosion Detection...")
        logger.info("="*50)

        try:
            model = self.create_mock_model()

            # Initialize gradient monitor
            gradient_monitor = GradientMonitor(
                config=GradientHealthConfig(
                    explosion_threshold=10.0,
                    vanishing_threshold=1e-7,
                    enable_clipping=True,
                    clip_value=1.0
                ),
                model=model
            )

            # Create exploding gradients
            for param in model.parameters():
                if param.grad is None:
                    param.grad = torch.randn_like(param) * 1000  # Large gradients

            # Check gradients
            is_healthy, stats = gradient_monitor.check_gradients(model)

            assert not is_healthy, "Exploding gradients should be unhealthy"
            assert any("explosion" in str(issue).lower() for issue in stats.get("issues", [])), \
                "Should detect gradient explosion"
            logger.info("✓ Gradient explosion detected")

            # Test clipping
            # First get the original norm without clipping
            original_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            original_norm_value = original_norm.item() if isinstance(original_norm, torch.Tensor) else original_norm

            # Reset gradients to original values for testing
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = torch.randn_like(param) * 1000  # Large gradients

            # Now clip the gradients
            clipped_norm = gradient_monitor.clip_gradients(model, max_norm=1.0)

            # Verify clipping happened
            final_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            final_norm_value = final_norm.item() if isinstance(final_norm, torch.Tensor) else final_norm

            assert final_norm_value <= 1.0 + 1e-6, "Gradients should be clipped to max_norm"
            logger.info(f"✓ Gradients clipped from {original_norm_value:.2f} to {final_norm_value:.2f}")

            self.test_results["gradient_explosion"] = "PASSED"

        except Exception as e:
            logger.error(f"Gradient explosion test failed: {e}")
            self.test_results["gradient_explosion"] = f"FAILED: {e}"

    def test_gradient_vanishing(self):
        """Test gradient vanishing detection."""
        logger.info("\n" + "="*50)
        logger.info("Testing Gradient Vanishing Detection...")
        logger.info("="*50)

        try:
            model = self.create_mock_model()

            # Initialize gradient monitor
            gradient_monitor = GradientMonitor(
                config=GradientHealthConfig(
                    explosion_threshold=100.0,
                    vanishing_threshold=1e-6
                ),
                model=model
            )

            # Create vanishing gradients
            for param in model.parameters():
                if param.grad is None:
                    param.grad = torch.randn_like(param) * 1e-8  # Tiny gradients

            # Check gradients
            is_healthy, stats = gradient_monitor.check_gradients(model)

            assert not is_healthy, "Vanishing gradients should be unhealthy"
            assert any("vanishing" in str(issue).lower() for issue in stats.get("issues", [])), \
                "Should detect gradient vanishing"
            logger.info("✓ Gradient vanishing detected")

            self.test_results["gradient_vanishing"] = "PASSED"

        except Exception as e:
            logger.error(f"Gradient vanishing test failed: {e}")
            self.test_results["gradient_vanishing"] = f"FAILED: {e}"

    def test_loss_divergence(self):
        """Test loss divergence detection."""
        logger.info("\n" + "="*50)
        logger.info("Testing Loss Divergence Detection...")
        logger.info("="*50)

        try:
            # Initialize loss validator
            loss_validator = LossValidator(
                config=LossValidationConfig(
                    divergence_threshold=50.0,  # Lower threshold for earlier detection
                    divergence_patience=2,  # Reduce patience for faster test
                    stability_window=5,
                    max_loss=1000.0  # Increase max_loss to avoid excessive loss warnings
                )
            )

            # Simulate diverging loss - start with more stable values to build history
            diverging_losses = [
                0.5, 0.6, 0.7, 0.8, 0.9,  # Initial stable phase
                1.0, 1.5, 2.5, 5.0, 10.0,  # Start diverging
                20.0, 50.0, 100.0, 200.0, 500.0  # Clear divergence
            ]

            divergence_detected = False
            for i, loss in enumerate(diverging_losses):
                is_valid, action, info = loss_validator.validate_loss(
                    torch.tensor(loss), step=i
                )

                if action == "restore_checkpoint":
                    divergence_detected = True
                    logger.info(f"✓ Divergence detected at step {i} with loss {loss:.2f}")
                    break

            assert divergence_detected, "Should detect loss divergence"

            self.test_results["loss_divergence"] = "PASSED"

        except Exception as e:
            logger.error(f"Loss divergence test failed: {e}")
            self.test_results["loss_divergence"] = f"FAILED: {e}"

    def test_reward_collapse(self):
        """Test DPO-specific reward margin collapse detection."""
        logger.info("\n" + "="*50)
        logger.info("Testing Reward Collapse Detection (DPO)...")
        logger.info("="*50)

        try:
            # For DPO, we need to check reward margin collapse
            # This happens when reward margins plateau near zero

            # Initialize reward validator (DPO-specific)
            reward_validator = LossValidator(
                config=LossValidationConfig(
                    plateau_threshold=0.01,  # Small threshold for near-zero detection
                    plateau_patience=5,  # Need some history to detect plateau
                    stability_window=5
                )
            )

            # Simulate reward margins that collapse to near zero and stay there
            # Start with healthy margins, then collapse to near zero
            reward_margins = [
                0.5, 0.4, 0.3, 0.2, 0.1,  # Decreasing but still healthy
                0.05, 0.02, 0.01,  # Approaching collapse
                0.005, 0.003, 0.002, 0.001,  # Collapsed margins
                0.001, 0.0008, 0.001, 0.0009  # Staying collapsed (plateau)
            ]

            collapse_detected = False
            for i, margin in enumerate(reward_margins):
                is_valid, action, info = reward_validator.validate_loss(
                    torch.tensor(margin), step=i
                )

                # Check if we've plateaued near zero (collapsed)
                if reward_validator.current_stats:
                    stats = reward_validator.current_stats
                    # Reward collapse = plateau near zero
                    if stats.is_plateaued and stats.mean < 0.01:
                        collapse_detected = True
                        logger.info(f"✓ Reward collapse detected at step {i}: "
                                  f"margin={margin:.4f}, mean={stats.mean:.4f}")
                        break

            assert collapse_detected, "Should detect reward margin collapse"

            self.test_results["reward_collapse"] = "PASSED"

        except Exception as e:
            logger.error(f"Reward collapse test failed: {e}")
            self.test_results["reward_collapse"] = f"FAILED: {e}"

    def test_state_recovery(self):
        """Test training state save and recovery."""
        logger.info("\n" + "="*50)
        logger.info("Testing State Recovery...")
        logger.info("="*50)

        try:
            # Initialize state manager
            state_manager = TrainingStateManager(
                checkpoint_dir=str(self.temp_dir / "states")
            )

            # Save a state
            from utils.training_state import TrainingState
            from datetime import datetime

            test_state = TrainingState(
                # Basic info
                stage="test",
                epoch=5.0,
                global_step=1000,
                best_metric=0.5,
                best_metric_name="loss",
                # Training progress
                total_epochs=10.0,
                total_steps=5000,
                samples_seen=32000,
                tokens_processed=1000000,
                # Model info
                model_name="test_model",
                checkpoint_path=str(self.temp_dir / "checkpoint"),
                adapter_config={"r": 16, "lora_alpha": 32},
                # Optimizer state
                learning_rate=1e-4,
                optimizer_state_dict=None,
                scheduler_state_dict=None,
                # Data state
                dataset_path="test_dataset",
                dataset_index=100,
                batch_size=32,
                gradient_accumulation_steps=4,
                # Metrics
                train_loss=0.5,
                eval_loss=0.6,
                metrics_history={"loss": [0.9, 0.7, 0.5]},
                # Recovery info
                timestamp=datetime.now().isoformat(),
                recovery_count=0,
                last_error=None,
                # Hardware state
                cuda_memory_allocated=1024*1024*1024,  # 1GB
                cuda_memory_reserved=2*1024*1024*1024  # 2GB
            )

            # Save state
            state_manager.save_state(test_state)
            logger.info("✓ State saved successfully")

            # Load state
            loaded_state = state_manager.load_state()

            assert loaded_state is not None, "Should load saved state"
            assert loaded_state.global_step == 1000, f"Step mismatch: {loaded_state.global_step}"
            assert loaded_state.best_metric == 0.5, f"Metric mismatch: {loaded_state.best_metric}"
            logger.info("✓ State loaded and verified successfully")

            self.test_results["state_recovery"] = "PASSED"

        except Exception as e:
            logger.error(f"State recovery test failed: {e}")
            self.test_results["state_recovery"] = f"FAILED: {e}"

    def test_error_handler_recovery(self):
        """Test central error handler's recovery mechanisms."""
        logger.info("\n" + "="*50)
        logger.info("Testing Error Handler Recovery...")
        logger.info("="*50)

        try:
            error_handler = ErrorHandler()

            # Test OOM error handling
            def simulate_oom():
                raise RuntimeError("CUDA out of memory")

            result = error_handler.safe_execute(
                simulate_oom,
                context={"test": "oom"}
            )

            assert result is None, "OOM should return None for recovery"
            assert error_handler.error_counts.get("RuntimeError", 0) > 0, \
                "Should track error count"
            logger.info("✓ OOM error caught and tracked")

            # Test successful execution
            def successful_func():
                return "success"

            result = error_handler.safe_execute(successful_func)
            assert result == "success", "Should execute successfully"
            logger.info("✓ Successful execution handled correctly")

            self.test_results["error_handler_recovery"] = "PASSED"

        except Exception as e:
            logger.error(f"Error handler recovery test failed: {e}")
            self.test_results["error_handler_recovery"] = f"FAILED: {e}"

    def test_precision_fallback(self):
        """Test precision handler fallback from fp16 to fp32."""
        logger.info("\n" + "="*50)
        logger.info("Testing Precision Fallback...")
        logger.info("="*50)

        try:
            # Initialize precision handler
            precision_handler = PrecisionHandler(
                config=PrecisionConfig(
                    use_bf16=False,
                    use_fp16=True,
                    use_fp32_fallback=True,
                    auto_detect_hardware=False  # Don't auto-detect for testing
                )
            )

            # Simulate overflow scenario
            if precision_handler.current_dtype == torch.float16:
                logger.info(f"✓ Starting with {precision_handler.current_dtype}")

                # Simulate multiple overflows
                for i in range(3):
                    precision_handler.consecutive_overflows += 1

                # Trigger fallback
                success = precision_handler.fallback_precision()

                assert success, "Fallback should succeed"
                assert precision_handler.current_dtype in [torch.bfloat16, torch.float32], \
                    f"Should fallback to bf16 or fp32, got {precision_handler.current_dtype}"
                logger.info(f"✓ Fell back to {precision_handler.current_dtype}")
            else:
                logger.info(f"✓ Precision handler initialized with {precision_handler.current_dtype}")

            self.test_results["precision_fallback"] = "PASSED"

        except Exception as e:
            logger.error(f"Precision fallback test failed: {e}")
            self.test_results["precision_fallback"] = f"FAILED: {e}"

    def run_all_tests(self):
        """Run all error handling tests."""
        logger.info("\n" + "="*60)
        logger.info("Starting Error Handling Test Suite")
        logger.info("="*60)

        # Run each test
        self.test_oom_recovery()
        self.test_nan_loss_handling()
        self.test_gradient_explosion()
        self.test_gradient_vanishing()
        self.test_loss_divergence()
        self.test_reward_collapse()
        self.test_state_recovery()
        self.test_error_handler_recovery()
        self.test_precision_fallback()

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("Test Results Summary")
        logger.info("="*60)

        passed = 0
        failed = 0

        for test_name, result in self.test_results.items():
            status = "✓ PASSED" if result == "PASSED" else "✗ FAILED"
            logger.info(f"{test_name:30s} {status}")

            if result == "PASSED":
                passed += 1
            else:
                failed += 1
                if result != "FAILED":
                    logger.error(f"  Error: {result}")

        logger.info("-"*60)
        logger.info(f"Total: {passed} passed, {failed} failed out of {passed + failed} tests")
        logger.info("="*60)

        # Cleanup
        self.cleanup()

        return failed == 0


def main():
    """Run the error handling test suite."""
    simulator = ErrorSimulator()

    try:
        success = simulator.run_all_tests()

        if success:
            logger.info("\n✅ All error handling tests passed!")
            exit(0)
        else:
            logger.error("\n❌ Some tests failed. Check the logs above.")
            exit(1)

    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
        simulator.cleanup()
        exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error during testing: {e}")
        simulator.cleanup()
        exit(1)


if __name__ == "__main__":
    main()