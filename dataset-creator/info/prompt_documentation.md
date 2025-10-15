# Prompt documentation 
The prompts in the prompts files all have embeddings which align with the personality traits and trait vectors. When the control function chooses a certain action, the embedded prompt vectors for that action are collected and a cosine similarity score is obtained for each prompt vector with the personality trait vector. The highest score is chosen as the prompt for the next action. 


### Prompt contextualization
Each prompt given to the models corresponds to an action chosen by the control function, and each action has a goal. In order to contextualize prompt selection, track deltas for each control feature (`cz`, `cg`, etc) and use these to define success of an action. 
If an action is successful (delta is positive above some threshold), add a factor that reduces the probability of it being selected for the next action by lowering its scalar value before it gets put into softmax (like 0.8x multiplier or something). This probability reduction is only effective for the immediate next turn.

### Prompt duplication prevention
Once a prompt has been used for one side in a particular negotiation dialog, it should be removed from the pool of available prompts for the next turn corresponding to that action to avoid repetition. 