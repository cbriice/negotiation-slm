# stage_5.md outlines the future APPLICATION of the training module being developed in this project.
The purpose of the training module is to train agents to optimize their behavior in this context, where they are the main players in the final stage of an ai agent negotiating software/network. Their behavior as part of the application is outlined below. 

**Stage 5 FSM flow/code outline:**

Every message is sent through REST endpoints with basic encryption for prototype. Connection is opened between agents off chain after sending on-chain confirmation pings. 

1. “handshake” exchange where each agent sends a basic identifying packet over the endpoint to verify each other’s identity before proceeding. Use intent vDID public key for verification, signed by root DID for confirmation that the right agent is representing this intent (”On Card Reception” section of security_spec.md)
2. Initiating (buy side) agent gets a prompt instance to create a strategy for negotiation. Context/prompt based on the information on the seller’s intent card, the user’s negotiation preferences and intent constraints. Output is JSON with “strategy” field (that’s it for now, but JSON format for functional expandability). Temp=0.5? Idk how “creative” I want it to be but also after fine tuning I could potentially have more headroom on temp 
    1. Sell side version of strategy creation probably just looks like the same but after receiving the first message from the buyer. Strategy would be more preloaded based on user’s preferences and constraints. 
    2. Strategy should be partially controlled by “knobs” that the SLM adjusts based on the strategy it decides on - target price, tone, risk tolerance, etc. This will help reduce the amount of inference that the second instance has to do on behavior based on the first instance’s strategy. Some coarse action indicator like “propose,” “counter,” “accept” and etc would also help guide action based on the strategy. Explicit definitions of the backbone of the strategy will reduce unexpected behavior
3. Once strategy is created, new instance of agent is called with a prompt to begin negotiation with general context about negotiating an intent representing the root user as well as the strategy generated in the previous step. This instance is instructed to generate a response using the strategy and that response is sent to the partner agent
    1. Prolog called to make sure initial offer is not illegal
4. From this point forward, a two-step process happens for both agents each time they receive a response from their partner:          
    1. First instance: SLM is prompted to analyze the response using the strategy which was created before negotiation starts and a list of things already tried. Outputs a concise plan for response generation. New plan is added to the running list of “things tried” underneath master strategy. Includes flags for requiring acknowledgement, etc
        1. Can edit “target_price”, “aggro”, “tone” fields based on new information. Edits “action_indicator” every time
    2. Second instance: SLM is prompted to generate a response based on the plan received from the first instance. Can guide response generation from the strategy based on user negotiation preferences - more aggressive, reasonable, etc. Also in prompt - most recent partner agent message
        1. Prolog called to make sure offer is legal
5. Negotiation ends
    1. If a deal is reached, initiator strategy instance would return simple offer command which response generation instance relays. Partner would return ACCEPT or something and negotiation ends.
    2. If there’s a stalemate, initiate drop-back mechanisms and restart 4. If no drop-backs are available for either agent, end here and return intents to the chain. 
    3. To define stalemate or force negotiation end, can observe how fast negotiations usually finish and adjust a max turns parameter if needed. 

Need to define how information flows between each stage of the mini-FSM that will control 4a → 4b. 

- minimal state should hold turn index, role, most important constraints, flags, recent actions from either party (offers)

Shit I’ll probably need:

- repetition guard - how do I tell if a stalemate is reached and they’re going in circles? Originally I just thought of an embed → similarity score from strategy to strategy to monitor how different each one is. However this could run into the problem where the SLM isn’t going in a loop, but just proposing slightly different things with mostly same conditions and this could set off false alarms. Probably some way to work around that problem in code so I think this is still potentially viable.