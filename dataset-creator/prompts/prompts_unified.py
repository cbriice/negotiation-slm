"""
prompts_unified.py
Has the single unified prompt for negotiation sims, updated every turn with conversation history and the next prompt for buyer/seller. In single-model generation, only this prompt is needed
Also contains general response prompts for forced conversation flow. 
"""

from utils.control_function import NegotiationState

# single model architecture system prompt
def get_unified_prompt(seed: dict, buyer_personality, seller_personality, conversation: dict, state: NegotiationState, action_prompt: str): 
    """
    Args: 
    seed: negotiation seed (unpacked from JSON format into python dict before passing into function)
    buyer_personality: buyer personality class, from prompts/personality_deck.py
    seller_personality: seller personality class
    conversation: conversation history dict (follows schema/conversation_json_template.py). passed into function after getting updated for each turn
    """

    # build conversation history
    dialog = conversation["dialog"]
    conversation_history = [f"{side}: {msg}" for side, msg in dialog]
    history_string = "\n".join(conversation_history)

    # since buyer always has first turn (turn index 0) you can deduce whose turn it is from turn counter in state
    turn_idx = state.r
    if turn_idx % 2 == 0:
        turn = "BUYER"
    else:
        turn = "SELLER"

    return f"""
        [ROLE] 
        You are a negotiation simulation model that generates synthetic dialogues for dataset creation. You will generate a dialog between Buyer and Seller based on descriptions of their personalities and prompts for each turn.
        
        [CONTEXT]
        TODO: seed information goes here, formatting not decided yet. item, info, context abt item

        Buyer: {buyer_personality.description}
        TODO: buyer info from seed here. price band, etc

        Seller: {seller_personality.description}
        TODO: seller info from seed here.

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
    buyer_clarify_respond: str = "Respond to the seller's query according to how the buyer's personality would respond."
    seller_clarify_respond: str = "Respond to the buyer's query according to how the seller's personality would respond."

    # when favorable conditions/personality factor decides to accept deal, use these prompts
    buyer_close_accept: str = "The seller has offered to close the deal at the current price. Generate a response to accept the deal according to how the buyer's personality would respond."
    seller_close_accept: str = "The buyer has offered to close the deal at the current price. Generate a response to accept the deal according to how the seller's personality would respond."

    # when rejecting an accept offer
    buyer_close_reject: str = "The seller has offered to close the deal at the current price. Generate a response to reject the deal according to how the buyer's personality would respond."
    seller_close_reject: str = "The buyer has offered to close the deal at the current price. Generate a response to reject the deal according to how the seller's personality would respond."