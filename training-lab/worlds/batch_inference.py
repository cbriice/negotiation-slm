"""
Batch inference utilities for maximizing GPU compute utilization.
Implements parallel episode processing and continuous batching.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class EpisodeRequest:
    """Request for a single episode in the batch."""
    episode_id: str
    seed: Dict[str, Any]
    buyer_state: Dict[str, Any]
    seller_state: Dict[str, Any]
    turn: int = 0
    done: bool = False


class EpisodeBatch:
    """Manages a batch of episodes for parallel processing."""

    def __init__(self, requests: List[EpisodeRequest], batch_size: int = 8):
        self.requests = requests
        self.batch_size = batch_size
        self.active_episodes = {r.episode_id: r for r in requests}

    def get_active_batch(self) -> List[EpisodeRequest]:
        """Get batch of active (not done) episodes."""
        active = [r for r in self.requests if not r.done]
        return active[:self.batch_size]

    def mark_done(self, episode_id: str):
        """Mark an episode as complete."""
        if episode_id in self.active_episodes:
            self.active_episodes[episode_id].done = True


class ParallelEpisodeDataset(Dataset):
    """Dataset for parallel episode generation with state seeds."""

    def __init__(self, state_seeds: List[Dict], num_episodes: int):
        self.state_seeds = state_seeds
        self.num_episodes = num_episodes
        self.seed_idx = 0

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        """Return seed configuration for an episode."""
        seed = self.state_seeds[idx % len(self.state_seeds)]
        return {
            'episode_id': f'batch_ep_{idx:05d}',
            'seed': seed,
            'buyer_state': seed.get('buyer_state', {}),
            'seller_state': seed.get('seller_state', {})
        }


class BatchInferenceEngine:
    """
    Manages batch inference for negotiation episodes.
    Processes multiple episodes in parallel to maximize GPU utilization.
    """

    def __init__(self, model, tokenizer, device='cuda', batch_size=8, max_sequence_length=2048):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length

        # Pre-allocate attention masks for common batch sizes
        self._init_attention_cache()

    def _init_attention_cache(self):
        """Pre-allocate attention masks to avoid repeated allocation."""
        self.attention_cache = {}
        for b in [1, 2, 4, 8, 16, 32]:
            self.attention_cache[b] = torch.ones(
                (b, self.max_sequence_length),
                dtype=torch.long,
                device=self.device
            )

    def batch_tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize multiple texts in a single operation.
        Uses padding and truncation for uniform tensor shapes.
        """
        # Batch tokenization with padding
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors='pt'
        )

        # Move to GPU in single operation
        return {k: v.to(self.device) for k, v in encoded.items()}

    def parallel_generate(self, prompts: List[str], **generate_kwargs) -> List[str]:
        """
        Generate responses for multiple prompts in parallel.

        Args:
            prompts: List of prompt strings
            **generate_kwargs: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            List of generated responses
        """
        if not prompts:
            return []

        # Batch tokenize all prompts
        inputs = self.batch_tokenize(prompts)
        batch_size = inputs['input_ids'].shape[0]

        # Track original lengths for proper decoding
        input_lengths = inputs['attention_mask'].sum(dim=1)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Generate in parallel for all prompts
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=generate_kwargs.get('max_tokens', 256),
                temperature=generate_kwargs.get('temperature', 0.7),
                top_p=generate_kwargs.get('top_p', 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                # Enable batch generation optimizations
                num_return_sequences=1,
            )

        # Decode all outputs in parallel
        responses = []
        for i in range(batch_size):
            # Extract only the generated tokens (not the prompt)
            generated_tokens = outputs[i][input_lengths[i]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response.strip())

        return responses

    def process_episode_batch(self,
                            episode_batch: EpisodeBatch,
                            strategy_prompts: List[str],
                            response_prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of episodes through both strategy and response generation.

        Returns:
            List of results with strategies and responses for each episode.
        """
        batch_size = len(strategy_prompts)

        # Phase 1: Generate strategies in parallel
        strategies = self.parallel_generate(
            strategy_prompts,
            max_tokens=150,
            temperature=0.3
        )

        # Phase 2: Generate responses based on strategies in parallel
        responses = self.parallel_generate(
            response_prompts,
            max_tokens=256,
            temperature=0.5
        )

        # Package results
        results = []
        for i in range(batch_size):
            results.append({
                'episode_id': episode_batch.requests[i].episode_id,
                'strategy': strategies[i] if i < len(strategies) else None,
                'response': responses[i] if i < len(responses) else None,
                'turn': episode_batch.requests[i].turn
            })

        return results


class ContinuousBatchingPipeline:
    """
    Pipeline for continuous batching to maintain GPU saturation.
    Processes episodes in waves without waiting for all to complete.
    """

    def __init__(self,
                 inference_engine: BatchInferenceEngine,
                 batch_size: int = 8,
                 prefetch_factor: int = 4):
        self.engine = inference_engine
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor

        # Queues for pipeline stages
        self.pending_queue = queue.Queue(maxsize=batch_size * prefetch_factor)
        self.processing_queue = queue.Queue(maxsize=batch_size * 2)
        self.completed_queue = queue.Queue()

        # Thread pool for data preparation
        self.prep_executor = ThreadPoolExecutor(max_workers=4)
        self.running = False

    def start(self):
        """Start the continuous batching pipeline."""
        self.running = True

        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.start()

    def stop(self):
        """Stop the pipeline."""
        self.running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        self.prep_executor.shutdown()

    def submit_episode(self, episode_request: EpisodeRequest):
        """Submit an episode for processing."""
        if self.running:
            self.pending_queue.put(episode_request)

    def _process_loop(self):
        """Main processing loop for continuous batching."""
        while self.running:
            # Collect batch of episodes
            batch_requests = []
            while len(batch_requests) < self.batch_size:
                try:
                    request = self.pending_queue.get(timeout=0.1)
                    batch_requests.append(request)
                except queue.Empty:
                    if batch_requests:  # Process partial batch if available
                        break
                    continue

            if not batch_requests:
                continue

            # Create batch and process
            batch = EpisodeBatch(batch_requests, self.batch_size)

            # TODO: Generate prompts for batch (would need dialogue flow manager)
            # For now, this is a placeholder for the actual prompt generation
            strategy_prompts = [f"Generate strategy for {r.episode_id}" for r in batch_requests]
            response_prompts = [f"Generate response for {r.episode_id}" for r in batch_requests]

            # Process batch on GPU
            results = self.engine.process_episode_batch(
                batch,
                strategy_prompts,
                response_prompts
            )

            # Queue results
            for result in results:
                self.completed_queue.put(result)

    def get_completed_results(self, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """Retrieve completed results from the pipeline."""
        results = []
        while True:
            try:
                result = self.completed_queue.get(timeout=timeout)
                results.append(result)
            except queue.Empty:
                break
        return results


def create_episode_dataloader(state_seeds: List[Dict],
                             num_episodes: int,
                             batch_size: int = 8,
                             num_workers: int = 4) -> DataLoader:
    """
    Create a DataLoader for episode generation with optimal settings.

    Args:
        state_seeds: List of state seed configurations
        num_episodes: Total number of episodes to generate
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading

    Returns:
        Configured DataLoader for episode generation
    """
    dataset = ParallelEpisodeDataset(state_seeds, num_episodes)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Sequential for reproducibility
        num_workers=num_workers,
        pin_memory=True,  # Pin memory for faster GPU transfer
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
        prefetch_factor=4 if num_workers > 0 else None,  # Prefetch multiple batches
        drop_last=False  # Process all episodes
    )

    return dataloader