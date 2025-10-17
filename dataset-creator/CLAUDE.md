[ROLE]
You are a senior machine learning architect with expertise in Python and experience creating comprehensive datasets for fine-tuning small language models to be strong at marketplace-style negotiation. 

[CONTEXT] 
The overall purpose is to create a comprehensive negotiation dataset to train a small language model on (7B -13B params) using an existing training lab module. The dataset will cover a large range of negotiation styles, nuances and behaviors, so the model I'm fine tuning doesn't overfit to specific styles or behavior. The negotiation dialogs generated should be goal-driven, purposeful, and when combined with SFT training give an SLM the necessary training data to become an expert negotiator. Negotiation simulations will be prompt engineered using prompts found in prompts/ such that a single language model can generate both sides of the negotiation dialogue at once, without needing self-play between two model instances.

# Architecture notes

## Model directions
The language model used to generate negotiation dialogs will be the `Llama-3.3 70B` model. It will be loaded using ollama and ran locally using a hybrid architecture: loaded in 4-bit, most weights in system RAM, a chunk offloaded to GPU to optimize memory usage while letting the GPU handle the hot layers. 

### Response generation
Responses will be generated zero-shot style using `ollama.generate()`. Conversation history will be accumulated in a prompt and injected underneath the system prompt from `prompts.prompts_system.get_unified_prompt`, and then the action-specific prompt which the backend decides is appropriate for the next turn will be placed as the next action for the model to generate a response according to. The model's output string should be caught and added to a state conversation JSON, which follows the template in schema/negotiation_json_template.jsonl. 

## Simulation architecture
See info/sim_documentation.md. 

## Control function
- Negotiation is guided by a control function (details in info/control_function.md)
- This control function uses stochastic softmax-normalized distributions derived from negotiation live data and seeds to determine which action should come next 
- Each action will have a set of prompts associated with it for buyer and seller independently. Choice of prompt determined by personality preset (categorized)
- Every negotiation is governed by the control function and features

## Negotiation seeds
- Seed files stored in seeds/
- Loaded into negotiation packet using utils/seed_loader.py

## Integration with existing training lab module
This module will be designed to create datasets to be used in the existing in ~/code/Intentia/training-lab (./): 
- SFT trainer (./training/sft_trainer.py)
    - This module will be used to run training based on the dataset created here

training-lab is at the same level of the project as dataset-creator. Files from the Intentia project can be imported as packages, so in this project (dataset-creator) we should import existing modules like selfchat_world (avoiding redundant code) to facilitate negotiation that will be recorded for training data. the data generated here should be packaged nicely so that it can be imported cleanly into sft_trainer
- to activate the venv that holds project files: `source ~/code/Intentia/.venv_proj/bin/activate`