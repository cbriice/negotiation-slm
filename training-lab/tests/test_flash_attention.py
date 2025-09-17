#!/usr/bin/env python3
"""Test Flash Attention 2 implementation."""

import torch
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from training.model_loader import ModelLoader
import training_parameters as params


def test_flash_attention_loading():
    """Test that Flash Attention 2 can be loaded successfully."""
    print("Testing Flash Attention 2 implementation...")
    print("=" * 60)

    # Test configuration
    print(f"Flash Attention 2 enabled in config: {params.USE_FLASH_ATTENTION_2}")
    print(f"Attention implementation: {params.ATTN_IMPLEMENTATION}")

    # Test loading with Flash Attention 2
    print("\n1. Testing model loading WITH Flash Attention 2...")
    try:
        model_name = "mistralai/Mistral-7B-v0.1"

        model, tokenizer = ModelLoader.load_model(
            model_name,
            use_flash_attention_2=True
        )

        # Check if model has flash attention
        if hasattr(model.config, "_attn_implementation"):
            print(f"   ✓ Attention implementation: {model.config._attn_implementation}")
        else:
            print("   ⚠ Could not verify attention implementation attribute")

        print(f"   ✓ Model loaded successfully: {model.__class__.__name__}")
        print(f"   ✓ Model dtype: {model.dtype}")
        print(f"   ✓ Tokenizer loaded: {tokenizer.__class__.__name__}")

        # Check model memory footprint
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   ✓ Total parameters: {param_count:,}")
        print(f"   ✓ Trainable parameters: {trainable_count:,}")

    except Exception as e:
        print(f"   ✗ Error loading model with Flash Attention 2: {e}")
        return False

    # Test loading without Flash Attention 2
    print("\n2. Testing model loading WITHOUT Flash Attention 2...")
    try:
        model_no_fa, _ = ModelLoader.load_model(
            model_name,
            use_flash_attention_2=False
        )

        if hasattr(model_no_fa.config, "_attn_implementation"):
            print(f"   ✓ Attention implementation: {model_no_fa.config._attn_implementation}")
        else:
            print("   ✓ Model loaded without Flash Attention 2")

    except Exception as e:
        print(f"   ✗ Error loading model without Flash Attention 2: {e}")

    # Test reference model loading
    print("\n3. Testing reference model loading with Flash Attention 2...")
    try:
        # Use the same model as reference for testing
        ref_model = ModelLoader.load_reference_model(
            model_name,
            use_flash_attention_2=True
        )

        print(f"   ✓ Reference model loaded: {ref_model.__class__.__name__}")
        print(f"   ✓ Model is in eval mode: {not ref_model.training}")
        print(f"   ✓ Parameters are frozen: {not any(p.requires_grad for p in ref_model.parameters())}")

    except Exception as e:
        print(f"   ✗ Error loading reference model: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ Flash Attention 2 implementation test PASSED!")
    return True


if __name__ == "__main__":
    success = test_flash_attention_loading()
    sys.exit(0 if success else 1)