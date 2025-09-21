"""
Test script to verify that response masking is working correctly.
Tests that labels are properly masked during training.
"""

import sys
import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.response_masking import (
    ResponseMaskingHandler,
    ResponseMaskingConfig,
    MultiFormatResponseMasker
)
import training_parameters as params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_response_masking_handler():
    """Test basic response masking functionality."""

    # Load a small tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create handler
    handler = ResponseMaskingHandler(tokenizer)

    # Test different format examples
    test_cases = [
        {
            "name": "Instruction-Response Format",
            "text": "### Instruction:\nWhat is 2+2?\n\n### Response:\nThe answer is 4.",
            "expected_masked": "### Instruction:\nWhat is 2+2?\n\n### Response:\n",
            "expected_supervised": "The answer is 4."
        },
        {
            "name": "Dialogue Format",
            "text": "User: Can you help me?\nAssistant: Yes, I'd be happy to help you.",
            "expected_masked": "User: Can you help me?\nAssistant: ",
            "expected_supervised": "Yes, I'd be happy to help you."
        },
        {
            "name": "Context-Response Format",
            "text": "### Context:\nState: {\"item\": \"laptop\", \"price\": 500}\nPartner: Will you take $400?\n\n### Response:\nI can do $450, that's my best offer.",
            "expected_masked": "### Context:\nState: {\"item\": \"laptop\", \"price\": 500}\nPartner: Will you take $400?\n\n### Response:\n",
            "expected_supervised": "I can do $450, that's my best offer."
        }
    ]

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {case['name']}")
        print(f"{'='*60}")

        # Tokenize
        input_ids = tokenizer.encode(case['text'], return_tensors='pt')[0]

        # Apply masking
        labels = handler.mask_labels_manual(input_ids, case['text'])

        # Validate
        validation = handler.validate_masking(input_ids, labels, case['text'])

        print(f"Total tokens: {validation['total_tokens']}")
        print(f"Masked tokens: {validation['masked_tokens']}")
        print(f"Supervised tokens: {validation['supervised_tokens']}")
        print(f"Masking ratio: {validation['masking_ratio']:.2%}")

        # Check supervised text
        if 'supervised_text_preview' in validation:
            print(f"Supervised text: {validation['supervised_text_preview'][:100]}")

            # Verify it matches expected
            if case['expected_supervised'] in validation['supervised_text_preview']:
                print("✓ Supervised text matches expected")
            else:
                print("✗ Supervised text mismatch!")
                print(f"  Expected: {case['expected_supervised']}")

        # Verify masked portion
        masked_ids = input_ids[labels == params.MASKING_IGNORE_INDEX]
        if len(masked_ids) > 0:
            masked_text = tokenizer.decode(masked_ids, skip_special_tokens=True)
            print(f"Masked text preview: {masked_text[:100]}")


def test_multi_format_masker():
    """Test multi-format response masker."""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create masker
    masker = MultiFormatResponseMasker(tokenizer)

    test_texts = [
        "### Instruction:\nExplain quantum computing\n\n### Response:\nQuantum computing uses quantum bits.",
        "User: Hello\nAssistant: Hi there! How can I help you today?",
        "### Context:\nNegotiation state\nPartner: Your offer?\n\n### Response:\nI propose $100."
    ]

    print(f"\n{'='*60}")
    print("Testing Multi-Format Masker")
    print(f"{'='*60}")

    for text in test_texts:
        format_name = masker.detect_format(text)
        print(f"\nDetected format: {format_name}")
        print(f"Text preview: {text[:50]}...")

        # Get appropriate collator
        collator = masker.get_collator_for_format(format_name)
        print(f"✓ Created collator for {format_name} format")


def test_collator_integration():
    """Test DataCollatorForCompletionOnlyLM integration."""

    from trl import DataCollatorForCompletionOnlyLM

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create handler
    handler = ResponseMaskingHandler(tokenizer)

    # Create collator
    collator = handler.create_completion_only_collator()

    print(f"\n{'='*60}")
    print("Testing DataCollator Integration")
    print(f"{'='*60}")

    # Test batch collation
    examples = [
        {"input_ids": tokenizer.encode("### Instruction:\nTest\n\n### Response:\nAnswer", return_tensors='pt')[0]},
        {"input_ids": tokenizer.encode("User: Hi\nAssistant: Hello", return_tensors='pt')[0]}
    ]

    try:
        # Collate batch
        batch = collator(examples)

        print(f"Batch keys: {batch.keys()}")
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")

        # Check masking
        for i, example in enumerate(examples):
            masked_count = (batch['labels'][i] == -100).sum().item()
            total_count = len(batch['labels'][i])
            print(f"Example {i+1}: {masked_count}/{total_count} tokens masked ({masked_count/total_count:.1%})")

        print("✓ Collator integration successful")

    except Exception as e:
        print(f"✗ Collator integration failed: {e}")


def test_response_positions():
    """Test finding response positions in tokenized input."""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

    # Create handler
    handler = ResponseMaskingHandler(tokenizer)

    print(f"\n{'='*60}")
    print("Testing Response Position Detection")
    print(f"{'='*60}")

    text = "### Instruction:\nFirst question\n\n### Response:\nFirst answer\n\n### Instruction:\nSecond question\n\n### Response:\nSecond answer"

    # Tokenize
    input_ids = tokenizer.encode(text, add_special_tokens=False)

    # Find positions
    positions = handler.find_response_positions(input_ids)

    print(f"Text: {text[:100]}...")
    print(f"Found {len(positions)} response positions: {positions}")

    # Verify positions
    for pos in positions:
        # Get text after position
        after_text = tokenizer.decode(input_ids[pos:min(pos+10, len(input_ids))])
        print(f"  Position {pos}: '{after_text[:30]}...'")


if __name__ == "__main__":
    print("="*60)
    print("Response Masking Test Suite")
    print("="*60)

    # Run tests
    test_response_masking_handler()
    test_multi_format_masker()
    test_collator_integration()
    test_response_positions()

    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")