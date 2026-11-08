# Sample data
I dug through old discord group chats and recovered this sample data that was generated from the pipeline, and some selfchat logs. 

One of the primary problems I encountered testing the trained model was that it learned to generate synthetic negotiations instead of 
actually negotiating. At the time I wasn't able to figure out why this happened and a lot of the iteration on masking / packet construction
were attempts to solve this problem. My diagnosis reflecting from my perspective now:

1. I originally split up the conversations into chunks so that the model would have varying input lengths, so that it didn't 
always predict the first or last utterance or etc. 
2. However, in a rush to iterate on training since data generation was slow I used the same negotiation for multiple packets with the 
model predicting the next utterance of the same negotiation up to 3 times per single conversation. 
3. The representation of the negotiation chunks that it saw spaced out in diff parts of training was too similar despite them 
not being presented together, so the model learned the pattern of constructing the whole dialog itself instead of the pattern of
predicting the next utterance.
4. At code level: `training-lab/evaluation/model_wrapper.py:171-173` makes `stop_sequences` hit `pass` so the model actually 
generates the entire packet and nothing cuts it off at the speaker label; no EOS after target turn meant the model didn't learn 
to stop talking after each utterance. 