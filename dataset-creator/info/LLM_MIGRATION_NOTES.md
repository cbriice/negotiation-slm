# LLM Interface Migration: Ollama to Hugging Face Transformers

## Overview
The LLM interface has been migrated from using Ollama with Llama models to using Hugging Face transformers with the Qwen2.5-14B-Instruct model.

## Key Changes

### Model Changes
- **Primary Model**: `Qwen/Qwen2.5-14B-Instruct` (14B parameters)
- **Fallback Model**: `Qwen/Qwen2.5-7B-Instruct` (7B parameters)
- **Loading Method**: Direct from Hugging Face Hub using transformers library

### Technical Improvements
1. **4-bit Quantization**: Enabled by default using bitsandbytes for memory efficiency
2. **Auto Device Mapping**: Automatically distributes model across available GPUs/CPU
3. **Native Transformers Integration**: Direct use of AutoModelForCausalLM and AutoTokenizer
4. **Better Memory Management**: Includes cleanup methods and GPU cache clearing

### Configuration Options
```python
config = LLMConfig(
    model="Qwen/Qwen2.5-14B-Instruct",
    fallback_model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.5,
    max_tokens=256,
    load_in_4bit=True,  # Enable 4-bit quantization
    device_map="auto",  # Auto device mapping
    use_gpu=True
)
```

## Installation

Install required dependencies:
```bash
pip install -r requirements.txt
```

Key dependencies:
- transformers>=4.36.0
- torch>=2.1.0
- accelerate>=0.25.0
- bitsandbytes>=0.41.0 (for 4-bit quantization)

## Usage Example

```python
from utils.llm_interface import LLMInterface, LLMConfig

# Initialize with default settings
llm = LLMInterface()

# Generate response
prompt = "Your negotiation prompt here..."
response = llm.generate(prompt, temperature=0.7, sim_id="sim_001")

# Check health
if llm.check_health():
    print("Model is ready")

# Clean up when done
llm.cleanup()
```

## Memory Requirements

With 4-bit quantization:
- **Qwen2.5-14B**: ~8-10GB VRAM
- **Qwen2.5-7B**: ~4-6GB VRAM

The models will automatically use CPU RAM if GPU memory is insufficient (with device_map="auto").

## Testing

Run the test script to verify the installation:
```bash
python test_llm_interface.py
```

## Exception Changes
- `OllamaConnectionError` → `ModelLoadError`
- `OllamaModelError` → `ModelInferenceError`

## Benefits of Migration
1. **No External Service Dependency**: No need to run Ollama service
2. **Better Control**: Direct control over model loading and inference
3. **4-bit Quantization**: Significantly reduced memory usage
4. **Flexible Device Management**: Automatic GPU/CPU distribution
5. **Native Stop Sequences**: Better handling of dialogue boundaries