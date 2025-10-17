# Prompt documentation 
The prompts in the prompts files all have embeddings which align with the personality traits and trait vectors. When the control function chooses a certain action, the embedded prompt vectors for that action are collected and a cosine similarity score is obtained for each prompt vector with the personality trait vector. The highest score is chosen as the prompt for the next action. 
All of the prompt files and the personality deck are in prompts/

### Prompt types and usage
**Basic actions**
  - price adjust, extras, clarify, deal close, deal accept, deal reject, general actions in prompts_unified.GeneralPrompts
**Personality-based actions**:
  - ultimatum. Ultimatums are only enabled for certain personalities based on their traits, dependent on parameter `eta`. 

**Flows**: 
  - price adjust, extras, clariy and close are self explanatory - based on control function outputs. 
  - Accept and reject are chosen based on the outline in sim_documentation.md, and this is triggered when one side sends a close offer. 
  - General actions: clarify respond would be triggered after clarify action is used by one side, and then the greeting prompts are used as the first two prompts by default for every negotiation. Third prompt in every negotiation (turn index 2) is chosen from price adjust pool as outlined in sim_documentation.md.

**Ultimatum**:
  - Ultimatum is a special case prompt category that is added to the pool of possible prompts when `eta` in range from personality card. Added to prompt pool when round pressure is high: `A_k` >= 0.7, and is in pool for price adjust as well as close. In the case that one side uses an ultimatum prompt, it's treated as a close proposal and the other side must respond with either accept or reject based on the outcome of random choice based on `p` (same outline in sim_documentation.md)

### Zero-shot generation prompt
The system prompt in prompts/prompts_unified.py will contain accumulated context of the conversation, current price and seed information that the `Llama-3.3-70B` will generate dialog according to using `/generate` in a zero-shot style, to maximize control of the model's behavior and keep context concise. The action-specific prompt is also injected into this system prompt. 
As negotiation progresses, each dialog output from the model is appended to the dialog dict contained in the negotiation packet dict. This will be unpacked and injected into the system prompt string every time get_unified_prompt is called.

### Prompt duplication prevention
Once a prompt has been used for one side in a particular negotiation dialog, it should be removed from the pool of available prompts for the next turn corresponding to that action to avoid repetition.