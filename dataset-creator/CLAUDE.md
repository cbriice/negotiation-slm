[ROLE]
You are a senior machine learning architect with expertise in Python and experience creating comprehensive datasets for fine-tuning small language models to be strong at marketplace-style negotiation. 

[CONTEXT]
You are programming a module for facilitating the creation of a comprehensive negotiation dataset in the style of the PACT negotiation dataset from the paper "We Argue to Agree" (see info/pact_info.md). 
The overall purpose is to create a comprehensive negotiation dataset to train a small language model on (7B -13B params) using an existing training lab module. The dataset will cover a large range of negotiation styles, nuances and behaviors, so the model I'm fine tuning doesn't overfit to specific styles or behavior. 

# Architecture notes

## Control function
- Negotiation is guided by a control function (details in info/control_function.md)
- This control function uses stochastic softmax-normalized distributions derived from negotiation live data and seeds to determine which action should come next 
- Each action will have a set of prompts associated with it for buyer and seller independently. Choice of prompt determined by personality preset (categorized)
- Every negotiation is governed by the control function and features

## Negotiation seeds
- Yet to be developed: I will provide a database of negotiation seeds as well as one for personality presets to choose from, with prompts for each action included. 

## Integration with existing training lab module
This module will be designed to create datasets to be used in the existing in ~/code/Intentia/training-lab (./): 
- SelfChatWorld (./worlds/selfchat_world.py, ./scripts/run_selfchat.py)
- SFT trainer (./training/sft_trainer.py)

training-lab is at the same level of the project as dataset-creator. Files from the Intentia project can be imported as packages, so in this project (dataset-creator) we should import existing modules like selfchat_world (avoiding redundant code) to facilitate negotiation that will be recorded for training data. the data generated here should be packaged nicely so that it can be imported cleanly into sft_trainer
- to activate the venv that holds project files: `source ~/code/Intentia/.venv_proj/bin/activate`