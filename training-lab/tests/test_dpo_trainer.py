"""Tests for DPO training module."""

import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock

from training.dpo_trainer import DPODataset, DPOTrainingModule
from utils.episode_logger import EpisodeLogger, TrainingStage


class TestDPODataset:
    """Test DPO dataset preparation."""

    def test_load_valid_dpo_pairs(self, tmp_path):
        """Test loading valid DPO pairs from NDJSON."""
        # Create test data
        test_data = [
            {
                "context": {
                    "state": {"item": "laptop", "price": 1000},
                    "recent_convo": [{"side": "buyer", "text": "How about $900?"}],
                    "partner_msg": "I can do $950"
                },
                "chosen": "That works for me. Let's close at $950.",
                "rejected": "No way, $800 or nothing."
            },
            {
                "context": {
                    "state": {"item": "phone", "price": 500},
                    "recent_convo": [],
                    "partner_msg": "What's your offer?"
                },
                "chosen": "I'd like to offer $450 for the phone.",
                "rejected": "$200 take it or leave it."
            }
        ]

        # Write to file
        data_file = tmp_path / "dpo_pairs.jsonl"
        with open(data_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        # Create mock tokenizer
        mock_tokenizer = Mock()

        # Load dataset
        dataset = DPODataset(str(data_file), mock_tokenizer)

        assert len(dataset) == 2
        assert dataset[0]['chosen'] == test_data[0]['chosen']
        assert dataset[0]['rejected'] == test_data[0]['rejected']

    def test_format_prompt(self, tmp_path):
        """Test prompt formatting from context."""
        # Create test data with context
        test_item = {
            "context": {
                "state": {"item": "laptop", "my_role": "seller", "target_price": 1000},
                "recent_convo": [
                    {"side": "buyer", "text": "What's your price?", "response": "I'm asking $1200"},
                    {"side": "buyer", "text": "Too high, how about $900?"}
                ],
                "partner_msg": "Can you do $950?"
            },
            "chosen": "I can meet you at $1000.",
            "rejected": "No, $1200 is final."
        }

        # Write to file
        data_file = tmp_path / "test.jsonl"
        with open(data_file, 'w') as f:
            f.write(json.dumps(test_item) + '\n')

        # Create dataset
        mock_tokenizer = Mock()
        dataset = DPODataset(str(data_file), mock_tokenizer)

        # Get formatted item
        item = dataset[0]
        prompt = item['prompt']

        # Check prompt contains expected elements
        assert "Negotiation State:" in prompt
        assert "laptop" in prompt
        assert "Recent Conversation:" in prompt
        assert "What's your price?" in prompt
        assert "Partner: Can you do $950?" in prompt

    def test_skip_invalid_entries(self, tmp_path):
        """Test that invalid entries are skipped."""
        # Create mixed valid/invalid data
        test_data = [
            {"context": {}, "chosen": "yes", "rejected": "no"},  # Valid
            {"context": {}, "chosen": "yes"},  # Missing rejected
            {"chosen": "yes", "rejected": "no"},  # Missing context
            {"context": {}, "chosen": "maybe", "rejected": "never"}  # Valid
        ]

        # Write to file
        data_file = tmp_path / "mixed.jsonl"
        with open(data_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        # Load dataset
        mock_tokenizer = Mock()
        dataset = DPODataset(str(data_file), mock_tokenizer)

        # Should only load valid entries
        assert len(dataset) == 2


class TestDPOTrainingModule:
    """Test DPO training module."""

    def test_initialization(self, tmp_path):
        """Test module initialization."""
        output_dir = tmp_path / "test_output"
        module = DPOTrainingModule(
            model_name="test-model",
            output_dir=str(output_dir)
        )

        assert module.model_name == "test-model"
        assert output_dir.exists()
        assert module.best_reward_margin == float('-inf')
        assert module.patience_counter == 0

    def test_prepare_dataset(self, tmp_path):
        """Test dataset preparation for HuggingFace format."""
        # Create test DPO pairs
        test_data = [
            {
                "context": {"state": {}, "partner_msg": "Test 1"},
                "chosen": "Good response 1",
                "rejected": "Bad response 1"
            },
            {
                "context": {"state": {}, "partner_msg": "Test 2"},
                "chosen": "Good response 2",
                "rejected": "Bad response 2"
            }
        ]

        data_file = tmp_path / "test_dpo.jsonl"
        with open(data_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        # Create module and mock tokenizer
        module = DPOTrainingModule(output_dir=str(tmp_path / "output"))
        mock_tokenizer = Mock()

        # Prepare dataset
        hf_dataset = module.prepare_dataset(str(data_file), mock_tokenizer)

        assert len(hf_dataset) == 2
        assert 'prompt' in hf_dataset.column_names
        assert 'chosen' in hf_dataset.column_names
        assert 'rejected' in hf_dataset.column_names

    def test_compute_metrics(self, tmp_path):
        """Test metrics computation."""
        module = DPOTrainingModule(output_dir=str(tmp_path / "output"))

        # Mock eval predictions
        mock_predictions = Mock()
        mock_predictions.predictions = [
            [0.8, 0.9, 0.7],  # Chosen rewards
            [0.3, 0.4, 0.2]   # Rejected rewards
        ]

        metrics = module.compute_metrics(mock_predictions)

        assert 'reward_margin' in metrics
        assert 'chosen_reward' in metrics
        assert 'rejected_reward' in metrics
        assert metrics['reward_margin'] == pytest.approx(0.5)  # 0.8 - 0.3
        assert metrics['chosen_reward'] == pytest.approx(0.8)
        assert metrics['rejected_reward'] == pytest.approx(0.3)

    def test_early_stopping_patience(self, tmp_path):
        """Test early stopping based on patience."""
        module = DPOTrainingModule(output_dir=str(tmp_path / "output"))
        module.early_stopping_patience = 3

        # Simulate no improvement
        module.patience_counter = 3
        assert module.should_early_stop() == True

        # Reset and test under threshold
        module.patience_counter = 2
        assert module.should_early_stop() == False

    def test_early_stopping_reward_collapse(self, tmp_path):
        """Test early stopping on reward collapse."""
        module = DPOTrainingModule(output_dir=str(tmp_path / "output"))

        # Simulate collapsed rewards
        module.reward_margins = [0.005, 0.003, 0.002, 0.001, 0.001]
        assert module.should_early_stop() == True

        # Test normal rewards
        module.reward_margins = [0.5, 0.4, 0.6, 0.5, 0.7]
        assert module.should_early_stop() == False

    @patch('training.dpo_trainer.ModelLoader')
    @patch('training.dpo_trainer.TRLDPOTrainer')
    def test_train_integration(self, mock_trainer_class, mock_loader, tmp_path):
        """Test training integration with mocked components."""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_ref_model = MagicMock()

        mock_loader.load_model.return_value = (mock_model, mock_tokenizer)
        mock_loader.load_reference_model.return_value = mock_ref_model

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        mock_result = MagicMock()
        mock_result.training_loss = 0.5
        mock_result.global_step = 100
        mock_result.metrics = {"train_runtime": 60}
        mock_trainer.train.return_value = mock_result

        # Create test data
        test_data = [
            {
                "context": {"state": {}, "partner_msg": "Test"},
                "chosen": "Good",
                "rejected": "Bad"
            }
        ]

        data_file = tmp_path / "train.jsonl"
        with open(data_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        # Create module and train
        module = DPOTrainingModule(
            model_name="test-model",
            output_dir=str(tmp_path / "output")
        )

        with patch('training.dpo_trainer.get_peft_model') as mock_peft:
            mock_peft.return_value = mock_model

            results = module.train(
                train_data_path=str(data_file),
                num_train_epochs=1
            )

        assert results['train_loss'] == 0.5
        assert results['total_steps'] == 100
        assert mock_trainer.train.called

    def test_train_from_episodes(self, tmp_path):
        """Test training from episodes with auto-generated pairs."""
        # Create test episodes
        episodes = [
            {
                "dialogue": [
                    {"text": "Test", "response": "Good response", "side": "agent"}
                ],
                "state": {"item": "test"},
                "meta": {"episode_id": "ep1"},
                "metrics": {"deal_closed": True, "protocol_violations": []}
            },
            {
                "dialogue": [
                    {"text": "Test", "response": "Bad response", "side": "agent"}
                ],
                "state": {"item": "test"},
                "meta": {"episode_id": "ep2"},
                "metrics": {"deal_closed": False, "protocol_violations": ["error"]}
            }
        ]

        episodes_file = tmp_path / "episodes.jsonl"
        with open(episodes_file, 'w') as f:
            for ep in episodes:
                f.write(json.dumps(ep) + '\n')

        # Mock the training method
        module = DPOTrainingModule(output_dir=str(tmp_path / "output"))

        with patch.object(module, 'train') as mock_train:
            mock_train.return_value = {"train_loss": 0.5}

            results = module.train_from_episodes(
                str(episodes_file),
                min_score_gap=0.2,
                num_train_epochs=1
            )

            # Check that train was called with generated DPO file
            assert mock_train.called
            call_args = mock_train.call_args[1]
            assert 'train_data_path' in call_args
            assert 'auto_dpo_pairs.jsonl' in call_args['train_data_path']


class TestIntegration:
    """Integration tests with existing pipeline."""

    def test_episode_logger_to_dpo_trainer(self, tmp_path):
        """Test full pipeline from EpisodeLogger to DPO training."""
        # Create episodes using EpisodeLogger
        logger = EpisodeLogger(
            base_dir=str(tmp_path / "data"),
            batch_size=10,
            stage=TrainingStage.SELFCHAT
        )

        # Create good and bad episodes
        good_episode = {
            "dialogue": [
                {"text": "What's your offer?", "response": "I can do $950", "side": "agent"}
            ],
            "state": {"item": "laptop", "price": 1000},
            "meta": {"episode_id": "good_1"},
            "metrics": {"deal_closed": True, "protocol_violations": [], "action_history": ["PROPOSE", "COMMIT", "ACK"]}
        }

        bad_episode = {
            "dialogue": [
                {"text": "What's your offer?", "response": "$500 final", "side": "agent"}
            ],
            "state": {"item": "laptop", "price": 1000},
            "meta": {"episode_id": "bad_1"},
            "metrics": {"deal_closed": False, "protocol_violations": ["lowball"], "action_history": ["PROPOSE", "NO_DEAL"]}
        }

        # Create DPO pair
        dpo_pair = logger.transform_to_dpo_pair(good_episode, bad_episode)
        assert dpo_pair is not None

        # Write DPO pair
        dpo_file = tmp_path / "dpo_pairs.jsonl"
        with open(dpo_file, 'w') as f:
            f.write(json.dumps(dpo_pair) + '\n')

        # Verify DPO trainer can load it
        mock_tokenizer = Mock()
        dataset = DPODataset(str(dpo_file), mock_tokenizer)
        assert len(dataset) == 1
        assert dataset[0]['chosen'] == "I can do $950"
        assert dataset[0]['rejected'] == "$500 final"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])