## Questions

1. Model and Infrastructure:
  - Will we be loading mistral-7b locally or via API? I see QLoRA parameters but need to confirm the model loading approach
  - Do you have GPU access for training? The parameters suggest CUDA support
  - Should we support distributed training across multiple GPUs?

  2. ParlAI Integration:
  - Should we use ParlAI's built-in model wrappers or create custom agents that interface with transformers/TRL directly?
  - For the RuleAgent in DialogPartnerWorld - should this be a scripted agent or another LLM with strict prompting?
  - How should we handle the ParlAI task inheritance for custom negotiation scenarios?

  3. Data Flow and Storage:
  - For episode logging - should we create separate directories for each training stage (baseline, self-chat, dialog-partner)?
  - The Prolog scoring mentioned - do you have an existing Prolog setup or should we implement rule-based scoring in Python?
  - For the Claude API protocol checking - is this a hard requirement or can we substitute with local evaluation?

  4. Training Pipeline:
  - Should the module support automatic progression through all 11 training stages or be manually controlled?
  - For DPO pair generation - what threshold/criteria should determine "clear separation" between good/bad examples?
  - How should we handle the "mutators" in step 9 (typos, jitters, etc.) - random or controlled perturbations?

  5. Architecture Decisions:
  - Single monolithic training script vs modular components (worlds/, agents/, tasks/, training/)?
  - Should we build on ParlAI's existing Deal-or-No-Deal and Casino tasks or reimplement?
  - Real-time metrics dashboard or just CSV logging?

# Answers

1.1: Locally
1.2: Yes - my local build has an rtx 5070ti GPU with 16GB VRAM and 64GB DDR5 RAM. 
1.3: No - this can be implemented if need be down the line but not for right now

2.1: Put training under TRL (SFT/DPO/ORPO with QLoRA) and only use ParlAI for data generation + evaluation. Wrap HF moddel as a very thin ParlAI `Agent` so it can "play" in SelfChatWorld and DialogPartnerWorld, but don't rely on ParlAI's built-in training wrappers for the fine=tuning passes. I.e. for the first built-in Deal or No Deal/Casino passes, load them via ParlAI to export/convert the data, then run a short SFT warm-start in TRL (instead of in ParlAI's loop). 
2.2: Scripted RuleAgent with randomized personas/tactics (including important permutations of rich conversation structure)
2.3: Create a tiny base JSONL teacher and inherit specialized teachers from it. Like: NegotiationJSONTeacher(FixedDialogTeacher) -> SelfPlayTeacher(NegotiationJSONTeacher); RuleOpponentTeacher(NegotiationJSONTeacher); HumanEvalTeacher(NegotiationJSONTeacher)

3.1: Yes. Organization always good
3.2: Python rule-based scoring is fine until training necessitates Prolog usage (decided later)
3.3: This is a hard requirement but it's for evaluating entire conversations from training in a more abstract sense. Only conversations which pass a preliminary local evaluation would go through to the LLM API module so that it doesn't waste tokens on junk. 

4.1: Manually controlled. Should have the ability to easily run any stage at any time, but generally it will be ran in order
4.2: Clear violations/out-of-line behavior vs. smooth examples that check all the boxes and reach a favorable agreement
4.3: Random within reason should be fine. Like there should be some reasoning behind the randomness but random is fine generally

5.1: Modular FOR SURE. Use the folders that exist to modulate
5.2: Use the existing tasks for the warm-start training, but data will be exported as JSONL and TRL SFTTrainer will be used to run initial warm-start training. This clarification is updated in step 2 of negotiation training plan in CLAUDE.md
5.3: a real-time metrics dashboard using Dash for the interface would be good. for numerical data, CSV logging is good. for logging and storing JSON data, NDJSON is good