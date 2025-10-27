# SFT packet schema doc
Per negotiation, two SFT packets will be generated: one making the buyer the assistant, and one vice versa. The side who is the assistant for a given packet is the side which the assistant_style system element is derived from. 

## field outlines
"messages" category contains the negotiation dialog loaded from the generated data. list obviously stretches beyond template for full dialog but just showed difference in opening structure for diff sides
user is opposite side in negotiation, assistant is model side

"system" category contains SFT tags, used so the model learns style, strategy application and context behind negotiation
    - assistant_style: derived from personality card
    - difficulty: derived from seed price difference - if buyer max much lower than seller min, high difficulty. if buyer max over seller min, low difficulty (and granularity in between extremes)
    - key_strategy: the central strategy used by assistant (determined by LLM review via Claude API call script)

"meta" category is backend-only, not used as input but for training bookkeeping

## schema
If assistant is buyer (they have first turn, if seller then first turn is user and second turn is assistant):
```json
{
    "messages": [
        {"role": "system", "content": "You are an expert negotiator."},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": ""}
    ],
    "system": {
        "assistant_style": "",
        "difficulty": 0.0,
        "key_strategy": ""
    },
    "meta": {
        "opponent_style": "",
        "favorable_outcome": "",
        "deal_closed": "",
        "rounds": 0
    }
}
```

## key_strategy options
  - anchor - assistant stands firm at one price and doesn't waver for anything, whether or not the deal is closed
  - compromise - assistant understands counterparty's position and works towards a compromise (problem-solving)
  - market_logic - assistant uses market context to persuade the counterparty
  - connection_appeal - assistant offers connections to friends/others as incentive for their offer
  - empathy_appeal - assistant appeals to empathy of counterparty, using personal situation as leverage
  - aggro - assistant aggressively attacks counterparty to get a favorable outcome
  - concede - assistant concedes to counterparty in order to close deal
  - walk_away - assistant quickly exits negotiation upon realizing a favorable deal can't be reached
  - flatter - assistant flatters the counterparty to get a more favorable deal

### prompt for claude api call script
You are a negotiation labeler labeling negotiations based on the tactic **MOST used** by the ASSISTANT during the negotiation.
You can choose from ONE of these tags:
  - **anchor** - assistant stands firm at one price and doesn't waver for anything, whether or not the deal is closed
  - **compromise** - assistant understands counterparty's position and works towards a compromise (problem-solving)
  - **market_logic** - assistant uses market context to persuade the counterparty
  - **connection_appeal** - assistant offers connections to friends/others as incentive for their offer
  - **empathy_appeal** - assistant appeals to empathy of counterparty, using personal situation as leverage
  - **aggro** - assistant aggressively attacks counterparty to get a favorable outcome
  - **concede** - assistant concedes to counterparty in order to close deal
  - **walk_away** - assistant quickly exits negotiation upon realizing a favorable deal can't be reached
  - **flatter** - assistant flatters the counterparty to get a more favorable deal
You must output ONE unformatted tag and nothing else. 
Negotiation: