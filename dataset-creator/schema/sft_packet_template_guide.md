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