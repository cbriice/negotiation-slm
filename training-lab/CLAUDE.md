# SLM negotiation agent training plan using ParlAI

You are a **senior machine learning engineer** with expertise in **Python and using ParlAI** to train SLMs. Your task is to code a ParlAI training module that uses Python (NOT CLI) to run tests and create training data for easily iteratable training of **mistral-7b** using QLoRA. Details for implementation of each and how to track metrics are below.

## This module will use **ParlAI** to train an SLM agent to be good at **negotiation** in the context of peer-to-peer marketplace style negotiations. From ParlAI it will use: 

Worlds:
1. SelfChatWorld for running seeded negotiation sims against itself.
2. DialogPartnerWorld - two modes, determined by `mode` flag: one for playing against a RuleAgent prompted with different negotiation profiles, the other for playing against a human via the console (ConsoleAgent).

Tasks (Negotiation Tasks, ParlAI built-in):
1. Deal or No Deal for teaching basic mechanics of negotaition behavior
2. Casino for getting a baseline of negotiation language

Teachers:
1. Teachers will inject custom STATE initial conditions as negotiation prompts for SelfChatWorld and DialogPartnerWorld simulations

---

## Negotiation training plan:

1. Run SelfChatWorld to get a baseline - how good is no-training model at negotiation? 
2. 0.5 epoch each of Deal or No Deal built-in task and Casino built-in task. Export tasks to JSONL following the SFT template and the model will be trained on a baseline SFT in TRL
    1. Goal: baseline training of SLM on negotiation behavior and proper dialogue for negotiations. Nudge it in the right direction before making training more specific. Save this as frozen reference for DPO: negotiation_baseline-version_number
3. Run SelfChatWorld (~1500 episodes). Save JSONL logs. Use minimal state seeds/opening lines with example intent metadata. 
    1. Note - maintain a subsection of the seeds used in this stage for regression control
	2. State seed template found in schema/state_template.jsonl
4. Annotate SelfChatWorld pass finding gold examples and shit examples for contrast. Use metadata tags for each exchange, style tokens for important (good) messages (keep these out of the prompt, only in metadata). Score each exchange using protocol/prolog - save stats and use to determine good/bad exchanges for pairing
5. Generate DPO pairs on clear violations vs clear good examples and train - reference stat analysis for determining good/bad examples
    1. Run short SFT pass on mutated gold examples (0.1-0.3 epoch, not every round)
6. Run DialogPartnerWorld (~2000 episodes, 50/50 split on side) with a RuleAgent to enforce tighter behavior restrictions and ability to terminate interaction if constraints violated. After baseline, use this to teach it more precise behavior without overfitting immediately on tight rule-guided runs
    1. RuleAgent’s persona should be randomized so that model doesn’t overfit to one opponent. Encode rich conversation structure into RuleAgent personas - embellishment, cheap talk, side offers, appeal to sympathy, world/external knowledge
7. Annotate/analyze DialogPartnerWorld chats, generate rich DPO packets and train
8. Re-run 3-7 with ~700 episodes total. Track statistics change - if some metrics are slipping, do a micro-RFT with top 10-20% clean examples (0.1-0.3 epoch, low LR)
9. Iterate 3-7 with mutators (price jitters, typos, bundle items, random conversation structure/tactics, time pressure, missing fields) with 3-5k “pairs” (~2000 episodes again). Store augmented examples separately so originals aren't overwritten. 
10. Annotate/analyze this data, generate DPO and train. If no significant change in performance observed, rerun with ORPO
11. Periodically run tests where agent negotiates against a human in DialogPartnerWorld to observe first-hand behaviors, reactions to extreme edge cases, etc. Log transcripts separately from training data to avoid contamination

Note - only generate DPO pairs when there’s clear separation between good and bad, otherwise route to KTO (thumbs up/down) or SFT

Negotiation actions: **IMPORTANT: TAGS SHOULD NOT BE INCLUDED IN AGENT RESPONSE!!!! TAGS ARE BACKEND ONLY!!!** [PROPOSE] for a normal bargaining turn. **For negotiation to end**, both sides must exchange a final [COMMIT] turn with the same **JSON-formatted TERMS_JSON object**, template in schema/terms_template.py. Other party sends [ACK] to accept and echoes the same TERMS_JSON (canonicalied JSON equivalence). [NACK] and return to [PROPOSE] action next if it doesn't accept - **explain why**, but don't send a new one back because negotiation shouldn't just turn into exchanging opposing JSON terms items. [NO_DEAL] is an option for an outright walk-away. **Only [COMMIT] + [ACK] ends negotiation successfully** - all other tags -> continue or failure.
    - NOTE: the agent messages **SHOULD NOT** contain the tag itself. The tag is just used on the backend to guide prompt injection for the agent to generate a response. I do not want the model to learn to put these tags before its responses

Training with ParlAI will be mostly negotiation-focused training but will have a strategy component too, since part of the agent’s task in the negotiation stage will be to generate a strategy to formulate a response based on. So it should be good at interpreting matched counterparties’ intents including the details relevant to negotiation to create a strategy that will help it gain some sort of advantage in principle.

Reference the **official ParlAI documentation** for syntax, arguments, etc. so that you don't hallucinate syntax errors.
---

## Strategy training plan:

Strategy JSON packets will be generated, then used in an SFT pass after negotiation fine-tuning. TODO: Develop a more detailed plan (this is just a rough idea)

Strategy template in schema/strategy_template.jsonl

---

## Metrics to track during simulations:

- protocol: constraint violations, protocol breaks (need to explicitly define protocol), verbosity, hallucinations
- negotiation meta: deals closed, % of terms met, #turns
- stats: per batch: tpb, ctpb, ltpb; per second: tps, ctps, ltps

Local agent metrics: verbosity, hallucination; Global agent metrics: token throughput; Teacher metrics: everything else. 

**Automated tracking**/scoring**:** Prolog to quantify constraint violations; scripts to quantify deals closed, terms met, num turns; stats are native to ParlAI; small-model Claude API call with custom prompt including behavior protocol to check protocol breaks during each interaction. 

- All stats should be numerical and some calculated per-episode, and then a session average for each should be computed. All of this data should be stored in CSVs in project files.
- Logs of every episode should be stored in JSONL format. These episodes will be used as training data via QLoRA for the SLM. Metadata should automatically be logged with each episode in a `"meta"` fied of the JSON packet.

**Metrics to calculate per-episode:** 
- "stats" section
- turns, token counts for agent responses, deal_closed (bool), % of terms met
- episode_id, dialogue_id, side, seed

## QLoRA usage:

Training will be done with **4-bit QLoRA** using **DPO pairs and SFT packets** generated from conversations from World episodes for training data. **IMPORTANT:** For parameters used when creating QLoRA training functions, **consult training_parameters.py FIRST** before coming up with your own parameters. Templates for SFT and DPO packets are in the schema/ folder. Note - 8-bit/nf4 can be toggled in training_parameters.py if gradient noise requires it.

