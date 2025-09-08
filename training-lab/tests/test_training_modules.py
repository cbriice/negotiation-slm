#!/usr/bin/env python3
"""
Test script for training modules functionality.
Tests data converters, scoring, and format conversions.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.converters.base import BaseConverter
from training.converters.sft_converter import SFTConverter
from training.converters.dpo_converter import DPOConverter
from training.converters.scoring import EpisodeScorer
from training.data_converter import DataConverter


class TestBaseConverter(unittest.TestCase):
    """Test BaseConverter functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.converter = BaseConverter(self.temp_dir)
    
    def test_save_load_jsonl(self):
        """Test JSONL save and load operations."""
        test_data = [
            {"id": 1, "text": "test1"},
            {"id": 2, "text": "test2"}
        ]
        
        test_file = Path(self.temp_dir) / "test.jsonl"
        self.converter.save_jsonl(test_data, test_file)
        
        loaded_data = self.converter.load_jsonl(test_file)
        self.assertEqual(len(loaded_data), 2)
        self.assertEqual(loaded_data[0]["text"], "test1")
    
    def test_extract_dialogue_history(self):
        """Test dialogue history extraction."""
        dialogue = [
            {"text": "Hello", "side": "buyer"},
            {"text": "Hi there", "side": "seller"},
            {"text": "I want to buy", "side": "buyer"},
            {"text": "What's your offer?", "side": "seller"}
        ]
        
        history = self.converter.extract_dialogue_history(dialogue, 3, window_size=2)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["text"], "Hi there")
        self.assertEqual(history[1]["text"], "I want to buy")


class TestSFTConverter(unittest.TestCase):
    """Test SFT conversion functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.converter = SFTConverter(self.temp_dir)
        
        # Create sample episode
        self.sample_episode = {
            "state": {
                "item": "laptop",
                "target_price": 500,
                "max_price": 600
            },
            "dialogue": [
                {
                    "text": "I'm interested in the laptop",
                    "response": "Great! It's in excellent condition.",
                    "side": "buyer",
                    "action": "PROPOSE"
                },
                {
                    "text": "What's your best price?",
                    "response": "I can offer it for $550",
                    "side": "seller",
                    "action": "PROPOSE"
                }
            ],
            "meta": {
                "episode_id": "test_001",
                "deal_closed": True,
                "side": "buyer"
            }
        }
    
    def test_episode_to_sft(self):
        """Test single episode conversion to SFT format."""
        sft_examples = self.converter.episode_to_sft(self.sample_episode)
        
        self.assertEqual(len(sft_examples), 2)
        
        # Check first example structure
        first_example = sft_examples[0]
        self.assertIn("input", first_example)
        self.assertIn("output", first_example)
        self.assertIn("meta", first_example)
        
        # Check input fields
        self.assertIn("state", first_example["input"])
        self.assertIn("recent_convo", first_example["input"])
        self.assertIn("partner_msg", first_example["input"])
        
        # Check output
        self.assertEqual(first_example["output"], "Great! It's in excellent condition.")
    
    def test_batch_convert(self):
        """Test batch conversion of episodes."""
        # Create test episodes file
        episodes_file = Path(self.temp_dir) / "test_episodes.jsonl"
        with open(episodes_file, 'w') as f:
            f.write(json.dumps(self.sample_episode) + '\n')
            f.write(json.dumps(self.sample_episode) + '\n')
        
        output_file = self.converter.batch_convert(str(episodes_file))
        
        self.assertTrue(Path(output_file).exists())
        
        # Load and verify converted data
        converted = self.converter.load_jsonl(Path(output_file))
        self.assertEqual(len(converted), 4)  # 2 episodes * 2 turns each


class TestDPOConverter(unittest.TestCase):
    """Test DPO conversion functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.converter = DPOConverter(self.temp_dir)
        
        # Create sample episodes
        self.good_episode = {
            "dialogue": [
                {
                    "text": "What's your offer?",
                    "response": "I can offer $500 for the item",
                    "side": "buyer"
                }
            ],
            "meta": {"quality_score": 9.0}
        }
        
        self.bad_episode = {
            "dialogue": [
                {
                    "text": "What's your offer?",
                    "response": "idk whatever",
                    "side": "buyer"
                }
            ],
            "meta": {"quality_score": 2.0}
        }
    
    def test_create_dpo_pair(self):
        """Test DPO pair creation."""
        dpo_pairs = self.converter.create_dpo_pair(
            self.good_episode, 
            self.bad_episode
        )
        
        self.assertGreater(len(dpo_pairs), 0)
        
        # Check first pair structure
        first_pair = dpo_pairs[0]
        self.assertIn("context", first_pair)
        self.assertIn("chosen", first_pair)
        self.assertIn("rejected", first_pair)
        self.assertEqual(first_pair["chosen"], "I can offer $500 for the item")
        self.assertEqual(first_pair["rejected"], "idk whatever")


class TestEpisodeScorer(unittest.TestCase):
    """Test episode scoring functionality."""
    
    def setUp(self):
        self.scorer = EpisodeScorer()
        
        self.sample_episode = {
            "dialogue": [
                {"text": "Hello", "response": "Hi", "action": "PROPOSE"},
                {"text": "Price?", "response": "$500", "action": "PROPOSE"},
                {"text": "Deal", "response": "[COMMIT]", "action": "COMMIT"}
            ],
            "meta": {
                "deal_closed": True,
                "num_turns": 3
            }
        }
    
    def test_score_episode(self):
        """Test episode scoring."""
        score = self.scorer.calculate_score(self.sample_episode)
        
        self.assertIsInstance(score, (int, float))
        # Score can be negative due to penalties
        self.assertGreater(score, -50)
        self.assertLess(score, 50)
    
    def test_classify_episode_quality(self):
        """Test quality classification."""
        quality = self.scorer.classify_episode_quality(self.sample_episode)
        self.assertIn(quality, ["excellent", "good", "fair", "poor"])
        # With deal_closed=True, should get a good score
        self.assertIn(quality, ["good", "excellent"])
    
    def test_rank_episodes(self):
        """Test episode ranking."""
        episodes = [
            {"dialogue": [], "meta": {"deal_closed": False}},
            self.sample_episode,
            {"dialogue": [{"text": "hi"}], "meta": {"deal_closed": True}}
        ]
        
        ranked = self.scorer.rank_episodes(episodes)
        
        # Check all episodes have scores
        for ep in ranked:
            self.assertIn("quality_score", ep)
        
        # Check they're sorted by score
        scores = [ep["quality_score"] for ep in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestDataConverter(unittest.TestCase):
    """Test main DataConverter orchestrator."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.converter = DataConverter(self.temp_dir)
        
        # Create sample episodes file
        self.episodes_file = Path(self.temp_dir) / "episodes.jsonl"
        sample_episode = {
            "state": {"item": "test"},
            "dialogue": [
                {
                    "text": "Hello",
                    "response": "Hi there",
                    "side": "buyer"
                }
            ],
            "meta": {"episode_id": "test_001"}
        }
        
        with open(self.episodes_file, 'w') as f:
            f.write(json.dumps(sample_episode) + '\n')
    
    def test_convert_episodes_to_sft(self):
        """Test SFT conversion through main converter."""
        output_file = self.converter.convert_episodes_to_sft(
            str(self.episodes_file)
        )
        
        self.assertTrue(Path(output_file).exists())
        
        # Verify content
        data = self.converter.sft_converter.load_jsonl(Path(output_file))
        self.assertGreater(len(data), 0)
    
    def test_score_episodes(self):
        """Test episode scoring through main converter."""
        output_file = self.converter.score_episodes(str(self.episodes_file))
        
        self.assertTrue(Path(output_file).exists())
        
        # Verify scored content
        data = self.converter.sft_converter.load_jsonl(Path(output_file))
        for episode in data:
            self.assertIn("quality_score", episode)
            self.assertIn("quality_tier", episode)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.converter = DataConverter(self.temp_dir)
        
        # Create realistic test episodes
        self.episodes = []
        for i in range(5):
            episode = {
                "state": {
                    "item": f"item_{i}",
                    "target_price": 100 * (i + 1),
                    "max_price": 150 * (i + 1)
                },
                "dialogue": [
                    {
                        "text": f"Interested in item_{i}",
                        "response": f"Yes, it's available for ${150*(i+1)}",
                        "side": "buyer",
                        "action": "PROPOSE"
                    },
                    {
                        "text": f"Can you do ${100*(i+1)}?",
                        "response": f"How about ${125*(i+1)}?",
                        "side": "seller",
                        "action": "PROPOSE"
                    }
                ],
                "meta": {
                    "episode_id": f"test_{i:03d}",
                    "deal_closed": i % 2 == 0,  # Half close deals
                    "num_turns": 2
                }
            }
            self.episodes.append(episode)
        
        # Save episodes
        self.episodes_file = Path(self.temp_dir) / "test_episodes.jsonl"
        with open(self.episodes_file, 'w') as f:
            for ep in self.episodes:
                f.write(json.dumps(ep) + '\n')
    
    def test_full_pipeline(self):
        """Test the complete data preparation pipeline."""
        results = self.converter.prepare_training_data(
            str(self.episodes_file)
        )
        
        # Check all expected outputs exist
        expected_keys = [
            'scored_episodes',
            'sft_all',
            'sft_high_quality',
            'dpo_pairs'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
            self.assertTrue(Path(results[key]).exists(), 
                          f"File {results[key]} should exist")
        
        # Verify scored episodes
        scored_data = self.converter.sft_converter.load_jsonl(
            Path(results['scored_episodes'])
        )
        self.assertEqual(len(scored_data), len(self.episodes))
        
        # Verify SFT data was created
        sft_data = self.converter.sft_converter.load_jsonl(
            Path(results['sft_all'])
        )
        self.assertGreater(len(sft_data), 0)


def run_tests(verbose=True):
    """Run all tests with optional verbosity."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestBaseConverter,
        TestSFTConverter,
        TestDPOConverter,
        TestEpisodeScorer,
        TestDataConverter,
        TestIntegration
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. See details above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test training modules")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("-t", "--test", type=str,
                       help="Run specific test class (e.g., TestSFTConverter)")
    
    args = parser.parse_args()
    
    if args.test:
        # Run specific test
        suite = unittest.TestSuite()
        suite.addTests(unittest.TestLoader().loadTestsFromName(args.test))
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        # Run all tests
        success = run_tests(args.verbose)
        sys.exit(0 if success else 1)