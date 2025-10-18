"""
Test script for the new Hugging Face transformers-based LLM interface.
"""

import sys
import torch
from utils.llm_interface import LLMInterface, LLMConfig

def test_llm_interface():
    """Test the LLM interface with Qwen model."""
    print("Testing LLM Interface with Hugging Face Transformers")
    print("=" * 50)

    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print("\nInitializing LLM interface...")

    # Create config with 4-bit quantization
    config = LLMConfig(
        model="Qwen/Qwen2.5-14B-Instruct",
        max_tokens=100,
        load_in_4bit=True,
        device_map="auto"
    )

    try:
        # Initialize interface
        llm = LLMInterface(config)
        print("✓ LLM interface initialized successfully")

        # Test health check
        if llm.check_health():
            print("✓ Health check passed")
        else:
            print("✗ Health check failed")
            return False

        # Test generation
        print("\nTesting text generation...")
        test_prompt = """You are a helpful assistant in a negotiation simulation.

BUYER: I'm interested in your product. What's your best price?

SELLER:"""

        response = llm.generate(test_prompt, 0.7, "test_001")

        print(f"\nGenerated response:\n{response}")
        print("\n✓ Generation test passed")

        # Get stats
        stats = llm.get_stats()
        print("\nUsage statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Cleanup
        print("\nCleaning up resources...")
        llm.cleanup()
        print("✓ Cleanup completed")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_llm_interface()
    sys.exit(0 if success else 1)