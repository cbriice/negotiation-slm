# Prompt documentation 
The prompts in the prompts files all have embeddings which align with the personality traits and trait vectors. When the control function chooses a certain action, the embedded prompt vectors for that action are collected and a cosine similarity score is obtained for each prompt vector with the personality trait vector. The highest score is chosen as the prompt for the next action. 
All of the prompt files and the personality deck are in prompts/

### Zero-shot generation prompt
The system prompt in prompts/prompts_unified.py will contain accumulated context of the conversation, current price and seed information that the `Llama-3.3-70B` will generate dialog according to using `/generate` in a zero-shot style, to maximize control of the model's behavior and keep context concise. The action-specific prompt is also injected into this system prompt. 
As negotiation progresses, each dialog output from the model is appended to the dialog dict contained in the negotiation packet dict. This will be unpacked and injected into the system prompt string every time get_unified_prompt is called.

### Prompt duplication prevention
Once a prompt has been used for one side in a particular negotiation dialog, it should be removed from the pool of available prompts for the next turn corresponding to that action to avoid repetition.