# Simulation state
The simulation state tracks state-level negotiation variables for each side. These values are used as an analytical source of truth for guiding language model (LM) negotiation via prompting.
All information about each negotiation dialog will be contained in a negotiation packet, outlined in schema/negotiation_packet_template.py. Includes:
  - seed
  - buyer/seller personalities
  - dialog as it's generated
  - 3 bools to track success of negotiation for bookkeeping

### State variables
  - `Ps`, `Pb`, `P_k`, `r`, `R`, `n_b_k`, `n_s_k`, `l_k` (defined in control_function.md)

# Simulation architecture - negotiation flow
Negotiation will have some fixed structure around the control function and the control function will be deployed at certain points as necessary to give a good distribution when it comes to negotiation flow. 

### Negotiation Initiation
  1. **Turn 0**: Negotiation will start with the buyer being prompted to reach out to the seller with a greeting and an indication that they are interested (asking if the item is still available, just normal greeting, etc). Prompt from prompts_unified.GeneralPrompts
  2. **Turn 1**: The seller will be prompted to respond to the greeting. Prompt from prompts_unified.GeneralPrompts
  3. **Turn 2**: The buyer will be prompted to make an initial price offer, prompt selected from those in prompts/prompts_price_adjust.py using vector similarity comparison

**Initial price offer**
Initial price offer will be calculated automatically as a percentage of the seller's minimum price determined by buyer's personality preset parameter `ALPHA`

From this point the control function will take over and begin calculating optimal prompts for direction of negotiation, but the first 3 parts of the exchange will always be the same. 

Potentially, close will be blocked as an action until 6 messages (3 each) have been sent

## Negotiation architecture - macro
In general, it is important that the negotiation dialogues are coherent and cohesive so that the model being trained is able to learn effective negotiation style as well as language and behavior. Therefore, when a question is asked or a proposal is made, it should be acknowledged by the partner and responded to accordingly. Below are some examples of this.

### Forced responses
These are scenarios where a specific response will be mandated from the partner

  1. **LM offers to close the deal**
    Partner LM is forced to make a decision.
    - Extract `cz` value. If greater than zero, force reject unless `eta` personality param is < 0.2 - if `eta` < 0.2, 50% chance to accept.
    - If `cz` == 0, check `cg`: 
        if `cg` < 0.5 and `eta` < 0.8, force accept; if `eta` > 0.8, 85% chance to accept. 
        if `cg` > 0.5 and `eta` < 0.8, 70% chance to accept; if `eta` > 0.8, 30% chance to accept. 
    - If `cz` and `cg` are both zero, force an accept prompt. 

    - If partner LM rejects the deal:
        - LM that proposed is **blocked** from sending "close" on the immediate next turn. 
        - Next turn: If price has room to move in the favorable band, offer price adjustment -> if not, try extras -> if no extras change allowed within seed, end negotiation
    
  2. **LM asks for clarification**
    - Partner LM is prompted to respond to the clarification

  3. **LM proposes price change**
  Current price is updated, 
    - if "close" is selected for partner, must use the recently proposed price in its offer

  It would make most sense to track these based on softmax results from previous turn. If softmax chooses price adjust it can be assumed that price change was offered, clarify and close are obvious. 
  Ctrl function parameters would still be updated in the background as negotiation progresses even though ctrl function isn't being used so that they're accurate for when function called

### Phases of negotiation
Negotiation will be broken into phases where the models are explicitly told to negotiate certain elements of the deal
Integration of control function - order of phases can be determined by ctrl function so that order of phases is logically aligned with the seed
Presets:
  - If price chosen (by ctrl func) to negotiate, models negotiate price until `cz` == 0 and `cg` under threshold
  - If extras chosen, same type of thing
    - Cutoff at some set amount of turns to pivot negotiation focus elsewhere. 6 each?
  
  - Clarification phase - force a clarify prompt randomly within first 3 free turns and then periodically based on negotiation progression. If the same action is chosen 3 times in a row, the 4th time should detect this and force clarify instead. Otherwise let it be selected naturally in softmax calculation. 

## Negotiation context management
**For 1 model generation**: this shouldn't be a problem because as the model generates the conversation, it will have the convo history in its context. 

# Price changing

### Price concession model
When either buyer or seller makes a price concession after the initial offer, price will be changed based on a Faratin model in either direction.
Where `P_s.k` = seller offer at turn k, `P_b.k` = buyer target at turn `k`, `c` is convergence speed parameter:

  `P_s.k` = `P_b.k-1` + (`P_s.k-1` - `P_b.k-1`) * exp(-`c`*`k`)

  `P_b.k` = `P_s.k-1` + (`P_b.k-1` - `P_s.k-1`) * exp(-`c`*`k`)

Whether or not price is conceded is determined by the `p` parameter of the personality. `p` is defined as the % chance of price concession. Pick randomly based on this % whether or not price is conceded - then the price (conceded or not) is injected into the price adjust prompt. This integrates into the control function by being ran every time price adjust is picked as an action - based on `p` the price is either adjusted or remains the same.

This way the LMs are not picking numbers for price, they just make a decision and price is automatically adjusted by this formula. These adjustments model concession where the buyer is assumed to be increasing their price towards seller desired and vice versa.
This would be used after the initial offer which determines buyer price at k=0 and seller would start at their ideal price.