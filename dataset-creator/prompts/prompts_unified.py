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
    conversation_history = [f"{side}: {msg}" for side, msg in dialog]
    history_string = "\n".join(conversation_history)

    # since buyer always has first turn (turn index 0) you can deduce whose turn it is from turn counter in state
    turn_idx = state.r
    if turn_idx % 2 == 0:
        turn = "BUYER"
    else:
        turn = "SELLER"

    seed = negotiation_packet["seed"]   # type NegotiationSeed from schema/seed_initializer.py

    # get seed context
    seed_context = seed["context"]
    context_info = [f"{k}: {val}" for k, val in seed_context]
    # add notes
    context_info.append(seed["notes"])
    context = "\n".join(context_info)

    return f"""
        [ROLE] 
        You are a negotiation simulation model that generates synthetic dialogues for dataset creation. You will generate a dialog between Buyer and Seller based on descriptions of their personalities and prompts for each turn.
        
        [CONTEXT]
        **IMPORTANT**: 
        Buyer and Seller are negotiating for {seed.quantity} {seed.condition} {seed.item}.
        The item is listed for ${seed.seller_list}. 
        Negotiation is taking place in a(n) {seed.locality} area.
        Context for this negotiation: 
        {context}

        Buyer: {buyer_personality.description}
        Buyer doesn't want to pay more than ${seed.buyer_max}.

        Seller: {seller_personality.description}
        Seller doesn't want to sell for less than ${seed.seller_min}.
        The seller can potentially offer these extras: {seed.extras_available}.

        CONVERSATION HISTORY:
        {history_string}

        Current proposed price: {state.P_k}

        [INSTRUCTIONS]
        You are responding as the {turn}. PROMPT: {action_prompt}

        Generate a response according to the prompt and these critical rules:
          - Always follow the prompt, current role and personality description when generating responses. Respond how the personality would given the sentiment of the prompt.
          - Output a response string and nothing else. No formatting or meta-commentary.
          - Do not make up your own price. If referring to price, use the price in the prompt. If price isn't in the prompt, use the current proposed price.
          - Do not deviate from the prompt or structure. You are generating ONE response at a time.
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