"""
prompts_unified.py
Has the single unified prompt for negotiation sims, updated every turn with conversation history and the next prompt for buyer/seller. In single-model generation, only this prompt is needed
Also contains general response prompts for forced conversation flow. 
"""

from utils.control_function import NegotiationState

# single model architecture system prompt
def get_unified_prompt(negotiation_packet: dict, state: NegotiationState, action_prompt: str): 
    """
    Args: 
    seed: negotiation seed (unpacked from JSON format into python dict before passing into function)
    buyer_personality: buyer personality class, from prompts/personality_deck.py
    seller_personality: seller personality class
    conversation: conversation history dict (follows schema/conversation_json_template.py). passed into function after getting updated for each turn
    """
    buyer_personality = negotiation_packet["buyer_personality"]
    seller_personality = negotiation_packet["seller_personality"]
    
    # build conversation history
    dialog = negotiation_packet["dialog"]
    conversation_history = "\n".join(
        [f"{side.upper()}: {msg.strip()}" for side, msg in dialog]
    )
    history_string = conversation_history

    # since buyer always has first turn (turn index 0) you can deduce whose turn it is from turn counter in state
    turn_idx = state.r
    if turn_idx % 2 == 0:
        turn = "BUYER"
    else:
        turn = "SELLER"

    seed = negotiation_packet["seed"]   # type NegotiationSeed from schema/seed_initializer.py
    category = seed.category

    # get seed context
    seed_context = seed.context
    context_info = [f"{k}: {val}" for k, val in seed_context.items()]
    # add notes
    if category == "vehicle":
        context_info.append("VEHICLE DETAILS: " + seed.notes)
    else:
        context_info.append(seed.notes)

    context = "\n".join(context_info)

    # get seed id
    id = seed.seed_id

    prompt = get_specific_prompt(seed, state, action_prompt, id, context, turn, history_string, seller_personality, buyer_personality)

    return prompt

# ---------------------------- get specific prompt depending on seed to adjust context of negotiation accordingly ---------------------------------

def get_specific_prompt(seed, state: NegotiationState, action_prompt: str, 
                        id: int, context: str, turn: str, history_string: str, 
                        seller_personality, buyer_personality) -> str:
    
    category = seed.category
    
    if category == "hospitality":     # hospitality specific prompt. extras are amenities, quantity is lease duration
        return f"""
        <|im_start|>system
        You are a negotiation simulation model that generates synthetic dialogues for dataset creation. You will generate a dialog between Buyer and Seller based on their personalities and prompts for each turn. 
        **IMPORTANT**: 
        Buyer and Seller are negotiating a {seed.quantity} lease for a {seed.condition} {seed.item}.
        List price: ${seed.seller_list} 
        Area: {seed.locality}
        Amenities: {seed.extras_available}.
        Context: 
        {context}

        BUYER PERSONALITY: {buyer_personality.description}
        Buyer max price: ${seed.buyer_max}.

        SELLER PERSONALITY: {seller_personality.description}
        Seller min price: ${seed.seller_min}.

        Current BUYER offer: ${state.buyer_offer}
        Current SELLER price: ${state.seller_offer}

        CONVERSATION HISTORY:
        {history_string}
        <|im_end|>
        <|im_start|>user
        [INSTRUCTIONS]
        You are the {turn}. PROMPT: {action_prompt}
        
        Generate a response according to the prompt and these critical rules:
          - Use natural, coherent language and phrasing following the PROMPT, current role and PERSONALITY when generating responses. 
          - Your output must contain only the response spoken by the {turn}. Do NOT include quotes, role labels, or any explanation.
          - If you mention price, use current proposed or price in prompt.
          - Never deviate from the PROMPT or structure. You are generating ONE response at a time.
        <|im_end|>
        <|im_start|>assistant
        """
    
    elif id > 3000 and id < 4000:   # service seeds. person to person tone should have slightly diff context imo. condition field meaningless as well
        return f"""
        <|im_start|>system
        You are a negotiation simulation model that generates synthetic dialogues for dataset creation. You will generate a dialog between Buyer and Seller based on their personalities and prompts for each turn. 
        **IMPORTANT**: 
        This negotiation is between two people for a service. It's professional, more subjective; focuses on cooperation, collaboration, persuasion, understanding.
        Buyer and Seller are negotiating for {seed.quantity} {seed.item}.
        List price: ${seed.seller_list} 
        Area: {seed.locality}
        Context: 
        {context}

        BUYER PERSONALITY: {buyer_personality.description}
        Buyer max price: ${seed.buyer_max}.

        SELLER PERSONALITY: {seller_personality.description}
        Seller min price: ${seed.seller_min}.
        Potential extras: {seed.extras_available}.

        Current BUYER offer: ${state.buyer_offer}
        Current SELLER price: ${state.seller_offer}

        CONVERSATION HISTORY:
        {history_string}
        <|im_end|>
        <|im_start|>user
        [INSTRUCTIONS]
        You are the {turn}. PROMPT: {action_prompt}
        
        Generate a response according to the prompt and these critical rules:
          - Use natural, coherent language and phrasing following the PROMPT, current role and PERSONALITY when generating responses. 
          - Your output must contain only the response spoken by the {turn}. Do NOT include quotes, role labels, or any explanation.
          - If you mention price, use current proposed or price in prompt.
          - Never deviate from the PROMPT or structure. You are generating ONE response at a time.
        <|im_end|>
        <|im_start|>assistant
        """
    
    else:               # return default prompt. for tech, vehicle seeds, other item based seeds
        return f"""
        <|im_start|>system
        You are a negotiation simulation model that generates synthetic dialogues for dataset creation. You will generate a dialog between Buyer and Seller based on their personalities and prompts for each turn. 
        **IMPORTANT**: 
        Buyer and Seller are negotiating for {seed.quantity} {seed.condition} {seed.item}.
        List price: ${seed.seller_list} 
        Area: {seed.locality}
        Context: 
        {context}

        BUYER PERSONALITY: {buyer_personality.description}
        Buyer max price: ${seed.buyer_max}.

        SELLER PERSONALITY: {seller_personality.description}
        Seller min price: ${seed.seller_min}.
        Potential extras: {seed.extras_available}.

        Current BUYER offer: ${state.buyer_offer}
        Current SELLER price: ${state.seller_offer}

        CONVERSATION HISTORY:
        {history_string}
        <|im_end|>
        <|im_start|>user
        [INSTRUCTIONS]
        You are the {turn}. PROMPT: {action_prompt}
        
        Generate a response according to the prompt and these critical rules:
          - Use natural, coherent language and phrasing following the PROMPT, current role and PERSONALITY when generating responses. 
          - Your output must contain only the response spoken by the {turn}. Do NOT include quotes, role labels, or any explanation.
          - If you mention price, use current proposed or price in prompt.
          - Never deviate from the PROMPT or structure. You are generating ONE response at a time.
        <|im_end|>
        <|im_start|>assistant
        """

# =============================================================================
# General response prompts for structured conversation flow
from dataclasses import dataclass

@dataclass
class GeneralPrompts:
    buyer_clarify_respond: str = "Respond to the seller's query according to how your personality would respond."
    seller_clarify_respond: str = "Respond to the buyer's query according to how your personality would respond."

    # opening dialog
    buyer_opener: str = "Greet the seller according to how your personality would."
    seller_opener: str = "Respond to the buyer's greeting according to how your personality would."