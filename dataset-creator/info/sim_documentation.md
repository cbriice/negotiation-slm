# Simulation state
The simulation state tracks state-level negotiation variables for each side. These values are used as an analytical source of truth for guiding language model (LM) negotiation via prompting.

### State variables
  - `Ps`, `Pb`, `P_k`, `r`, `R` (defined in control_function.md)

### State computations
The negotiation state for each side will have different values and be updated differently based on personality 

A way to structure this in code could be that u define each personality as its own class that inherits a base negotiator class and assigns parameter/prompt variables based on personality data.

# Simulation architecture - negotiation flow
Negotiation will have some fixed structure around the control function and the control function will be deployed at certain points as necessary to give a good distribution when it comes to negotiation flow. 

### Negotiation Initiation
  1. Negotiation will start with the buyer being prompted to reach out to the seller with a greeting and an indication that they are interested (asking if the item is still available, just normal greeting, etc)
  2. The seller will be prompted to respond to the greeting
  3. The buyer will be prompted to make an initial price offer (prompts for personality deck specifics will determine how this offer goes)

**Initial price offer**
Initial price offer will be calculated automatically as a percentage of the seller's minimum price determined by buyer's personality preset parameter `ALPHA`

From this point the control function will take over and begin calculating optimal prompts for direction of negotiation, but the first 3 parts of the exchange will always be the same. 

Potentially, close will be blocked as an action until 6 messages (3 each) have been sent

### Price concession model
When either buyer or seller makes a price concession after the initial offer, price will be changed based on a Faratin model in either direction.
Where `P_s.k` = seller offer at turn k, `P_b.k` = buyer target at turn `k`, `c` is convergence speed parameter:

  `P_s.k` = `P_b.k-1` + (`P_s.k-1` - `P_b.k-1`) * exp(-`c`*`k`)

  `P_b.k` = `P_s.k-1` + (`P_b.k-1` - `P_s.k-1`) * exp(-`c`*`k`)

This way the LMs are not picking numbers for price, they just make a decision and price is automatically adjusted by this formula. These adjustments model concession where the buyer is assumed to be increasing their price towards seller desired and vice versa.
This would be used after the initial offer which determines buyer price at k=0 and seller would start at their ideal price.

### Forced responses
These are scenarios where a specific response will be mandated from the partner

  1. **LM offers to close the deal**
    - Partner LM is forced to make a decision
    - Extract `cz` value. If greater than zero, will mostly reject but some personalities may accept anyways
    - If `cz` == 0, check `cg`: prompt to accept or reject based on favorability of `cg` value
    - If `cz` and `cg` are both zero, force an accept prompt. 

    - If partner LM rejects the deal:
        - LM that proposed is **blocked** from sending "close" on the immediate next turn. 
        - Next turn: If price has room to move in the favorable band, offer price adjustment -> if not, try extras -> if no extras change allowed within seed, end negotiation
    
  2. **LM asks for clarification**
    - Partner LM is prompted to respond to the clarification

  3. **LM proposes price change**
    - if "clarify" is selected for partner, must acknowledge price change in clarification ask. like "before i accept, what about..." or sm
    - if "close" is selected partner, must use the recently proposed price in its offer

  It would make most sense to track these based on softmax results from previous turn. If softmax chooses finetune or compensate it can be assumed that price change was offered, clarify and close are obvious. 
  Ctrl function parameters would still be updated in the background as negotiation progresses even though ctrl function isn't being used so that they're accurate for when function called

### Phases of negotiation
Negotiation will be broken into phases where the models are explicitly told to negotiate certain elements of the deal
Integration of control function - order of phases can be determined by ctrl function so that order of phases is logically aligned with the seed
Presets:
  - If price chosen (by ctrl func) to negotiate, models negotiate price until `cz` == 0 and `cg` under threshold
  - If extras chosen, same type of thing
  - During price negotiation, if progress not rly being made towards ideal, personality becomes more accentuated in prompts (more aggressive, adding extra items, tactics, etc)
    - Cutoff at some set amount of turns to pivot negotiation focus elsewhere. 6 each? Some personalities could have prompts chosen to ddd
  
  - Clarification phase - force a clarify prompt periodically so shit doesn't just proceed super fast to close based only on price and measurables.

### Prompt contextualization
Each prompt given to the models corresponds to an action chosen by the control function, and each action has a goal. In order to contextualize prompt selection, track deltas for each control feature (`cz`, `cg`, etc) and use these to define success of an action. If an action is successful (delta is positive above some threshold), add a factor that reduces the probability of it being selected for the next action by lowering its scalar value before it gets put into softmax. 