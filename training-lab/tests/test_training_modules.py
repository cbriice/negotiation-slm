#!/usr/bin/env python3
"""
Test script for training modules functionality.
Tests DataConverter with EpisodeLogger integration.
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

from training.data_converter import DataConverter
from training.converters.builtin_export import BuiltinTaskExporter
from utils.episode_logger import EpisodeLogger, TrainingStage
from utils.protocol_validator import calculate_protocol_score


class TestEpisodeLogger(unittest.TestCase):
    """Test EpisodeLogger functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.logger = EpisodeLogger(base_dir=self.temp_dir, batch_size=10)
    
    def test_load_write_episodes(self):
        """Test episode load and write operations."""
        test_episodes = [
            {"dialogue": [{"text": "test1", "side": "agent"}], "state": {}, "meta": {}},
            {"dialogue": [{"text": "test2", "side": "partner"}], "state": {}, "meta": {}}
        ]
        
        test_file = Path(self.temp_dir) / "test.jsonl"
        self.logger.write_episodes(test_episodes, test_file)
        
        loaded_episodes = self.logger.load_episodes(test_file)
        self.assertEqual(len(loaded_episodes), 2)
        self.assertEqual(loaded_episodes[0]["dialogue"][0]["text"], "test1")
    
    def test_transform_to_sft_packet(self):
        """Test SFT packet transformation."""
        episode = {
            "dialogue": [
                {"text": "Hello", "side": "partner", "response": "Hi there"},
                {"text": "Want to trade?", "side": "partner", "response": "Sure"},
            ],
            "state": {"item": "laptop", "price": 500},
            "meta": {"episode_id": "test_001"}
        }
        
        sft_packets = self.logger.transform_to_sft_packet(episode)
        
        self.assertEqual(len(sft_packets), 2)
        self.assertIn("input", sft_packets[0])
        self.assertIn("output", sft_packets[0])
        self.assertEqual(sft_packets[0]["output"], "Hi there")
    
    def test_batch_transform_to_sft(self):
        """Test batch SFT transformation."""
        episodes = [
            {
                "dialogue": [{"text": "test", "response": "response"}],
                "state": {},
                "meta": {},
                "metrics": {}
            }
        ]
        
        sft_packets = self.logger.batch_transform_to_sft(episodes)
        self.assertIsInstance(sft_packets, list)
    
    def test_filter_by_quality(self):
        """Test quality filtering."""
        episodes = [
            {"metrics": {"protocol_violations": [], "action_history": ["PROPOSE"], "deal_closed": True}},
            {"metrics": {"protocol_violations": ["leaked_tags"], "action_history": [], "deal_closed": False}}
        ]
        
        filtered = self.logger.filter_by_quality(episodes, min_score=0.8)
        self.assertEqual(len(filtered), 1)


class TestDataConverter(unittest.TestCase):
    """Test DataConverter with EpisodeLogger integration."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.converter = DataConverter(self.temp_dir)
        
        # Create sample episodes file
        self.sample_episodes = [
            {
                "dialogue": [
                    {"text": "Hello", "response": "Hi", "side": "agent"},
                    {"text": "Want to trade?", "response": "Sure", "side": "partner"}
                ],
                "state": {"item": "laptop"},
                "meta": {"episode_id": "001"},
                "metrics": {
                    "protocol_violations": [],
                    "action_history": ["PROPOSE", "PROPOSE"],
                    "deal_closed": True
                }
            },
            {
                "dialogue": [
                    {"text": "Test", "response": "Reply", "side": "partner"},
                    {"text": "Another turn", "response": "OK", "side": "agent"}
                ],
                "state": {"item": "phone"},
                "meta": {"episode_id": "002"},
                "metrics": {
                    "protocol_violations": ["leaked_tags"],
                    "action_history": ["PROPOSE", "PROPOSE"],
                    "deal_closed": False
                }
            }
        ]
        
        self.episodes_file = Path(self.temp_dir) / "test_episodes.jsonl"
        with open(self.episodes_file, 'w') as f:
            for episode in self.sample_episodes:
                f.write(json.dumps(episode) + '\n')
    
    def test_convert_episodes_to_sft(self):
        """Test SFT conversion through DataConverter."""
        output_file = self.converter.convert_episodes_to_sft(
            str(self.episodes_file)
        )
        
        self.assertTrue(Path(output_file).exists())
        
        # Load and verify
        with open(output_file, 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
    
    def test_score_episodes(self):
        """Test episode scoring."""
        output_file = self.converter.score_episodes(str(self.episodes_file))
        
        self.assertTrue(Path(output_file).exists())
        
        # Load and verify scoring
        episodes = self.converter.episode_logger.load_episodes(output_file)
        self.assertEqual(len(episodes), 2)
        
        # Check protocol scores were added
        for episode in episodes:
            self.assertIn("protocol_score", episode["meta"])
            self.assertIn("quality_tier", episode["meta"])
    
    def test_auto_generate_dpo(self):
        """Test auto DPO generation."""
        # Need episodes with score differences
        output_file = self.converter.auto_generate_dpo(
            str(self.episodes_file),
            score_threshold=0.3
        )
        
        self.assertTrue(Path(output_file).exists())


class TestBuiltinTaskExporter(unittest.TestCase):
    """Test BuiltinTaskExporter functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = BuiltinTaskExporter(self.temp_dir)
    
    def test_parse_dealnodeal(self):
        """Test Deal-or-No-Deal parsing."""
        text = """context: book=3 book_value=1 hat=2 hat_value=5
dialogue:
YOU: I'd like 2 books
THEM: I need the hat"""
        labels = ["Sure, take them"]
        
        dialogue, state = self.exporter._parse_dealnodeal(text, labels)
        
        self.assertEqual(len(dialogue), 2)
        self.assertIn("items", state)
        self.assertIn("book", state["items"])
        self.assertEqual(dialogue[-1]["response"], "Sure, take them")
    
    def test_save_load_jsonl(self):
        """Test JSONL operations."""
        test_data = [{"id": 1}, {"id": 2}]
        test_file = Path(self.temp_dir) / "test.jsonl"
        
        self.exporter.save_jsonl(test_data, test_file)
        loaded = self.exporter.load_jsonl(test_file)
        
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["id"], 1)


if __name__ == "__main__":
    unittest.main()