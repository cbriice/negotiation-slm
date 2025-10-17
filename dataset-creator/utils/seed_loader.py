"""
Seed loading module for negotiation simulations.

This module handles loading negotiation seeds from JSON/JSONL files and provides
various methods for seed selection and iteration.
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Union, Iterator
from schema.seed_initializer import NegotiationSeed


class SeedLoader:
    """Loads and manages negotiation seeds from JSON/JSONL files."""

    def __init__(self, seed_file: Optional[Union[str, Path]] = None):
        """
        Initialize the seed loader.

        Args:
            seed_file: Path to the seed file (JSON or JSONL format)
        """
        self.seeds: List[dict] = []
        self.seed_objects: List[NegotiationSeed] = []
        self.current_index = 0

        if seed_file:
            self.load_seeds(seed_file)

    def load_seeds(self, seed_file: Union[str, Path]) -> None:
        """
        Load seeds from a JSON or JSONL file.

        Args:
            seed_file: Path to the seed file

        Raises:
            FileNotFoundError: If the seed file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        seed_path = Path(seed_file)

        if not seed_path.exists():
            raise FileNotFoundError(f"Seed file not found: {seed_path}")

        with open(seed_path, 'r') as f:
            content = f.read().strip()

            # Try to parse as regular JSON (array format like tech_seeds.jsonl)
            try:
                seeds_data = json.loads(content)
                if isinstance(seeds_data, list):
                    self.seeds = seeds_data
                else:
                    # Single object, wrap in list
                    self.seeds = [seeds_data]
            except json.JSONDecodeError:
                # Try JSONL format (one JSON object per line)
                self.seeds = []
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    if line.strip():  # Skip empty lines
                        try:
                            self.seeds.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            raise json.JSONDecodeError(
                                f"Invalid JSON on line {line_num}: {e.msg}",
                                e.doc, e.pos
                            )

        # Convert to NegotiationSeed objects
        self.seed_objects = [NegotiationSeed(seed) for seed in self.seeds]
        print(f"Loaded {len(self.seeds)} seeds from {seed_path}")

    def get_seed_by_id(self, seed_id: int) -> Optional[NegotiationSeed]:
        """
        Get a specific seed by its ID.

        Args:
            seed_id: The ID of the seed to retrieve

        Returns:
            NegotiationSeed object or None if not found
        """
        for seed_obj in self.seed_objects:
            if seed_obj.seed_id == seed_id:
                return seed_obj
        return None

    def get_random_seed(self) -> Optional[NegotiationSeed]:
        """
        Get a random seed from the loaded seeds.

        Returns:
            Random NegotiationSeed object or None if no seeds loaded
        """
        if not self.seed_objects:
            return None
        return random.choice(self.seed_objects)

    def get_next_seed(self) -> Optional[NegotiationSeed]:
        """
        Get the next seed in sequential order (cycles back to start).

        Returns:
            Next NegotiationSeed object or None if no seeds loaded
        """
        if not self.seed_objects:
            return None

        seed = self.seed_objects[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.seed_objects)
        return seed

    def get_all_seeds(self) -> List[NegotiationSeed]:
        """
        Get all loaded seeds.

        Returns:
            List of all NegotiationSeed objects
        """
        return self.seed_objects.copy()

    def iter_seeds(self) -> Iterator[NegotiationSeed]:
        """
        Iterate through all seeds.

        Yields:
            NegotiationSeed objects one at a time
        """
        for seed in self.seed_objects:
            yield seed

    def filter_by_category(self, category: str) -> List[NegotiationSeed]:
        """
        Filter seeds by category.

        Args:
            category: Category to filter by

        Returns:
            List of NegotiationSeed objects matching the category
        """
        return [s for s in self.seed_objects if s.category == category]

    def get_categories(self) -> List[str]:
        """
        Get unique categories from loaded seeds.

        Returns:
            List of unique category names
        """
        return list(set(s.category for s in self.seed_objects))

    def reset_index(self) -> None:
        """Reset the current index for sequential access."""
        self.current_index = 0

    def __len__(self) -> int:
        """Return the number of loaded seeds."""
        return len(self.seed_objects)

    def __repr__(self) -> str:
        """String representation of the loader state."""
        return f"SeedLoader(seeds={len(self.seeds)}, categories={self.get_categories()})"


class MultiFileSeedLoader:
    """Manages loading seeds from multiple category files."""

    def __init__(self, seed_directory: Union[str, Path] = "seeds"):
        """
        Initialize multi-file seed loader.

        Args:
            seed_directory: Directory containing seed files
        """
        self.seed_dir = Path(seed_directory)
        self.loaders: dict[str, SeedLoader] = {}
        self.all_seeds: List[NegotiationSeed] = []

    def load_all_seeds(self, pattern: str = "*.jsonl") -> None:
        """
        Load all seed files matching the pattern.

        Args:
            pattern: Glob pattern for seed files (default: *.jsonl)
        """
        if not self.seed_dir.exists():
            raise FileNotFoundError(f"Seed directory not found: {self.seed_dir}")

        seed_files = list(self.seed_dir.glob(pattern))

        for seed_file in seed_files:
            # Extract category name from filename (e.g., tech_seeds.jsonl -> tech)
            category_name = seed_file.stem.replace("_seeds", "")

            loader = SeedLoader(seed_file)
            self.loaders[category_name] = loader
            self.all_seeds.extend(loader.get_all_seeds())

        print(f"Loaded {len(self.all_seeds)} total seeds from {len(seed_files)} files")

    def get_random_seed(self, category: Optional[str] = None) -> Optional[NegotiationSeed]:
        """
        Get a random seed, optionally from a specific category.

        Args:
            category: Optional category to select from

        Returns:
            Random NegotiationSeed object or None if no seeds available
        """
        if category and category in self.loaders:
            return self.loaders[category].get_random_seed()
        elif self.all_seeds:
            return random.choice(self.all_seeds)
        return None

    def get_loader_for_category(self, category: str) -> Optional[SeedLoader]:
        """
        Get the SeedLoader for a specific category.

        Args:
            category: Category name

        Returns:
            SeedLoader instance or None if category not found
        """
        return self.loaders.get(category)

    def get_all_categories(self) -> List[str]:
        """Get list of all loaded categories."""
        return list(self.loaders.keys())

    def __repr__(self) -> str:
        """String representation."""
        return f"MultiFileSeedLoader(categories={self.get_all_categories()}, total_seeds={len(self.all_seeds)})"


# Convenience functions for CLI usage
def load_seed_file(filepath: str) -> SeedLoader:
    """
    Load a single seed file.

    Args:
        filepath: Path to the seed file

    Returns:
        Configured SeedLoader instance
    """
    return SeedLoader(filepath)


def load_seed_directory(dirpath: str = "seeds") -> MultiFileSeedLoader:
    """
    Load all seed files from a directory.

    Args:
        dirpath: Path to directory containing seed files

    Returns:
        Configured MultiFileSeedLoader instance
    """
    loader = MultiFileSeedLoader(dirpath)
    loader.load_all_seeds()
    return loader


if __name__ == "__main__":
    # Example usage for testing
    import argparse

    parser = argparse.ArgumentParser(description="Test seed loader functionality")
    parser.add_argument("--file", type=str, help="Path to seed file")
    parser.add_argument("--dir", type=str, default="seeds", help="Path to seed directory")
    parser.add_argument("--id", type=int, help="Get specific seed by ID")
    parser.add_argument("--random", action="store_true", help="Get random seed")
    parser.add_argument("--all", action="store_true", help="List all seeds")
    parser.add_argument("--category", type=str, help="Filter by category")

    args = parser.parse_args()

    if args.file:
        # Load single file
        loader = load_seed_file(args.file)

        if args.id:
            seed = loader.get_seed_by_id(args.id)
            if seed:
                print(f"Seed {args.id}: {seed.item} - ${seed.seller_list}")
            else:
                print(f"Seed {args.id} not found")
        elif args.random:
            seed = loader.get_random_seed()
            if seed:
                print(f"Random seed: {seed.item} (ID: {seed.seed_id})")
        elif args.all:
            for seed in loader.iter_seeds():
                print(f"  ID {seed.seed_id}: {seed.item} - ${seed.seller_list}")
    else:
        # Load directory
        loader = load_seed_directory(args.dir)
        print(f"Loaded categories: {loader.get_all_categories()}")

        if args.category:
            seed = loader.get_random_seed(args.category)
            if seed:
                print(f"Random {args.category} seed: {seed.item} (ID: {seed.seed_id})")
        elif args.random:
            seed = loader.get_random_seed()
            if seed:
                print(f"Random seed: {seed.item} (ID: {seed.seed_id})")