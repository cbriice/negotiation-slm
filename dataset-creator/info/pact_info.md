## This is a summary document for relevant information about the PACT dataset which is being used as inspiration for the creation of this custom dataset. 

**What it is / domain**: Synthetic, LLM-generated negotiation dialogues between a traveler and a travel agent in the tourism sector, built to support personality-driven, argumentation-based negotiation research. 

**Scale & stats**: 8,687 conversations, 186,241 turns (avg 21.43 turns per convo), ~11.5M tokens; min turns 8, max 24. Useful for long multi-turn behavior. 

**Annotation schema (per turn)**: ABN-specific dialog acts (e.g., negotiate-price-increase/decrease/nochange, negotiate-add-X/remove-X, ask/ tell/ justify/ assure price or X, accept/consent, etc.) with frequencies reported; this enables act-conditioned modeling and evaluation. 

**Personalities**: Three personality profile families attached to interlocutors—Argumentation Profile, Preference Profile, Buying Style Profile—so models can condition on persona traits during act prediction/response generation. 

**Generation pipeline**: 3-stage LLM pipeline—(1) Generation, (2) Filtering (to remove empties/repeats/off-topic/toxicity; ~quality curation), (3) Quality assessment (auto + manual). Built with Gemini-1.5-Flash prompts; curation emphasizes argumentation structure and persona consistency. 

**Grounding / knowledge**: RAG over a curated tour packages database (amenities, services, prices) to reduce hallucination and keep price/option mentions factual; retrieval via BERT embeddings + cosine similarity. 

**Tasks supported (for training/eval splits)**:
PR (Personality Recognition), DAP (Dialog Act Prediction), RG (Response Generation), with a standard 70/20/10 train/val/test split and metrics for each (Acc/F1 for PR/DAP; BLEU/METEOR/ROUGE/DIST/BERTScore + persona/act accuracy for RG). 

**Intended use**: Fine-tune LMs to produce persona-consistent, act-aligned, fact-grounded negotiation turns in multi-turn settings; paper reports that PACT-fine-tuned models outperform baselines on PR/DAP/RG. 

**Name collision note**: There’s also a separate GitHub project named PACT (Pairwise Auction Conversation Testbed) focused on 20-round buyer/seller bidding with private values. Don’t confuse it with the tourism PACT above.