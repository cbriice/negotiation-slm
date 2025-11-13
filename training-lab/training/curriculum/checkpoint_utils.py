"""
Checkpoint utilities for curriculum training.
Handles saving and loading phase checkpoints with LoRA adapters.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


def save_phase_checkpoint(
    model,
    phase_num: int,
    phase_name: str,
    step: int,
    total_phases: int,
    output_dir: Path
) -> Path:
    """
    Save LoRA adapter checkpoint at phase boundary.

    Args:
        model: Model with LoRA adapters to save
        phase_num: Phase number (1-indexed)
        phase_name: Name of the phase
        step: Current training step
        total_phases: Total number of phases in curriculum
        output_dir: Base output directory for checkpoints

    Returns:
        Path to saved checkpoint directory
    """
    checkpoint_dir = output_dir / f"phase_{phase_num}_step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter weights
    logger.info(f"Saving LoRA adapters to {checkpoint_dir}")
    model.save_pretrained(str(checkpoint_dir))

    # Save metadata
    metadata = {
        "phase_number": phase_num,
        "phase_name": phase_name,
        "step": step,
        "total_phases": total_phases,
        "timestamp": datetime.now().isoformat()
    }

    metadata_path = checkpoint_dir / "checkpoint_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Phase {phase_num} checkpoint saved successfully")
    return checkpoint_dir


def load_phase_checkpoint(checkpoint_dir: str) -> Tuple[Path, Dict[str, Any]]:
    """
    Load phase checkpoint and validate.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Tuple of (adapter_path, metadata_dict)

    Raises:
        FileNotFoundError: If checkpoint or metadata missing
        ValueError: If checkpoint structure invalid
    """
    checkpoint_path = Path(checkpoint_dir)

    # Validate checkpoint directory exists
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Check for required files
    adapter_model = checkpoint_path / "adapter_model.safetensors"
    adapter_config = checkpoint_path / "adapter_config.json"
    metadata_file = checkpoint_path / "checkpoint_metadata.json"

    if not adapter_model.exists():
        raise FileNotFoundError(f"Adapter model not found: {adapter_model}")
    if not adapter_config.exists():
        raise FileNotFoundError(f"Adapter config not found: {adapter_config}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_file}")

    # Load and validate metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    required_fields = ["phase_number", "phase_name", "step", "total_phases"]
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Missing required metadata field: {field}")

    logger.info(f"Loaded checkpoint from phase {metadata['phase_number']} (step {metadata['step']})")
    return checkpoint_path, metadata


def adjust_phase_ranges(phases: List[Dict], start_from_phase_num: int) -> List[Dict]:
    """
    Adjust phase step ranges to start from 0.

    Args:
        phases: List of phase configurations
        start_from_phase_num: Phase number to start from (1-indexed)

    Returns:
        Filtered and adjusted list of phases

    Raises:
        ValueError: If phase number invalid
    """
    if start_from_phase_num < 1 or start_from_phase_num > len(phases):
        raise ValueError(
            f"Invalid phase number {start_from_phase_num}. "
            f"Must be between 1 and {len(phases)}"
        )

    # Convert to 0-indexed
    start_phase_idx = start_from_phase_num - 1

    # Get the offset (starting step of the resume phase)
    offset = phases[start_phase_idx]["step_start"]

    # Filter to keep only phases from start_from_phase onward
    filtered_phases = phases[start_phase_idx:]

    # Shift all step ranges down by offset
    adjusted_phases = []
    for phase in filtered_phases:
        adjusted_phase = phase.copy()
        adjusted_phase["step_start"] = phase["step_start"] - offset
        adjusted_phase["step_end"] = phase["step_end"] - offset
        adjusted_phases.append(adjusted_phase)

    logger.info(f"Adjusted {len(adjusted_phases)} phases to start from step 0")
    return adjusted_phases


def print_resume_banner(
    checkpoint_path: Path,
    metadata: Dict[str, Any],
    adjusted_phases: List[Dict]
):
    """
    Print clear banner showing resume information.

    Args:
        checkpoint_path: Path to loaded checkpoint
        metadata: Checkpoint metadata
        adjusted_phases: Adjusted phase configurations
    """
    phase_ranges = ", ".join([
        f"Phase {i+metadata['phase_number']} ({p['step_start']}-{p['step_end']})"
        for i, p in enumerate(adjusted_phases)
    ])

    banner = f"""
{'='*80}
RESUMING CURRICULUM TRAINING
{'='*80}
Loaded adapters from: {checkpoint_path}
Previous training completed through step {metadata['step']} (end of phase {metadata['phase_number']})
Resuming with phases {metadata['phase_number']+1}-{metadata['total_phases']}, step counter reset to 0
New step ranges: {phase_ranges}
{'='*80}
"""
    print(banner)
    logger.info("Resume configuration loaded successfully")


def validate_phase_compatibility(
    checkpoint_metadata: Dict[str, Any],
    current_phases: List[Dict],
    start_from_phase_num: int
) -> None:
    """
    Validate that current curriculum is compatible with checkpoint.

    Args:
        checkpoint_metadata: Metadata from checkpoint
        current_phases: Current phase configuration
        start_from_phase_num: Phase to resume from

    Raises:
        Warning if configurations appear incompatible
    """
    # Check if number of phases changed
    if len(current_phases) != checkpoint_metadata["total_phases"]:
        logger.warning(
            f"Phase count mismatch: checkpoint has {checkpoint_metadata['total_phases']} phases, "
            f"but current curriculum has {len(current_phases)} phases"
        )

    # Check if starting from a phase beyond what was trained
    if start_from_phase_num <= checkpoint_metadata["phase_number"]:
        logger.warning(
            f"Resuming from phase {start_from_phase_num}, but checkpoint completed phase "
            f"{checkpoint_metadata['phase_number']}. This will re-train already completed phases."
        )
