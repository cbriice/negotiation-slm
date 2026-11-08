# negotiation-slm

An end-to-end pipeline for teaching a 7B language model to negotiate marketplace deals:

1. **`dataset-creator/`** generates synthetic buyer/seller negotiations with a locally run
   Qwen2.5-14B-Instruct. A mathematical **control function** picks each turn's action, a
   **personality deck** shapes style and tactics, and a concession model sets prices, so the
   LLM only writes the words.
2. **`training-lab/`** fine-tunes Mistral-7B with **QLoRA** using a three-phase **curriculum**
   and **weighted next-utterance loss masking**. A self-chat harness then evaluates the
   resulting adapters.

**Note:**
I had Claude move this out of a private monorepo to this separate one that I could make public and have on
my GitHub profile. 

I built this in 2025 as the negotiation-model workstream of an old startup idea, a project in which AI
agents negotiate on behalf of their users. I designed the system: the control function and
simulation flow, the personality/prompt-selection scheme, the data format, and the
curriculum and masking strategy. Most of the code was written with Claude Code under my direction.

Unfortunately I don't have any record of the data generated or the test runs besides what I dug up from old discord 
chats in samples/. I kept them out of GitHub because the files were large. During a training run that spiked my GPU 
power above capacity, my PSU blew and it corrupted the WSL instance I stored the files on, making them unrecoverable. 
From memory I did not get many successful self-play trials as a result of training, and I was in the process of diagnosing 
and tuning the system when we decided to pivot the underlying concept for MVP development. I never finished the project fully to 
get successful demonstrable results. I remember encountering catastrophic forgetting in some cases and also during 
one specific test run Mistral spontaneously started outputting tokens in Korean for some reason. Some data was 
recovered from git history in training-lab/evaluation/notes/degenerate_responses.txt. It was a very interesting 
sub-project and taught me a lot about small model harnesses, inference and LoRA training. 

---

```
 seeds (item, reservation prices, context)          personality deck (12 trait profiles)
                    │                                              │
                    └──────────────► simulation_runner ◄──────────┘
                                          │  each turn:
                                          │   control function → action (price / extra / close / clarify)
                                          │   cosine(prompt traits, personality traits) → prompt
                                          │   Faratin concession model → price
                                          │   Qwen2.5-14B (4-bit) → one utterance
                                          ▼
                              negotiation runs (JSON) ──► sft_converter ──► SFT packets
                                                             (one buyer-as-assistant and one
                                                              seller-as-assistant per negotiation)
                                          │
 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                                          ▼
 messages_to_sft ──► reformat_to_next_utterance ──► pre_tokenize ──► train_curriculum (QLoRA)
                     (2–3 prefixes per conversation,               phase 1 → 2 → 3: data mix and
                      only the final assistant turn                 loss weights shift toward
                      is the target)                                negotiation-only targets
                                                                          │
                                                                          ▼
                                                            run_selfchat / run_interactive
```

## Dataset generation (`dataset-creator/`)

Each negotiation starts from a **seed**: an item, its condition, the seller's list and minimum
prices, the buyer's maximum, the available extras, and free-form context such as market type,
urgency or information advantage (`schema/seed_template.py`, example in
`seeds/examples/tech_seeds.json`). The seed template calls for 15–20% of seeds to have no
price overlap between buyer and seller, which makes them the hardest cases.

Buyer and seller each get one of **12 personalities** (`prompts/personality_deck.py`):
straightforward, desperate, skeptic, bluffer, confrontational, haggler, emotional conceder,
rationalizer, stonewaller, opportunist, haughty and stubborn. Each personality is a 15-trait
vector (assertiveness, dominance, cooperativeness, deceptiveness, patience and so on), and its
behavioral parameters derive from those traits:

| Parameter | Meaning | Definition |
|---|---|---|
| `p` | chance of conceding price on a price-adjust turn | mean(1−confidence, cooperativeness, flexibility, 1−persistence) |
| `c` | Faratin convergence speed | √(cooperativeness · flexibility) |
| `eta` | if >0.5 and round pressure A_k >= 0.7, enable ultimatums | √(assertiveness · dominance · risk tolerance) |
| `beta` | price threshold adjustment weight | per personality |
| `alpha` | buyer only: for first offer, % value of seller minimum | per personality |

At every free turn the **control function** (`utils/control_function.py`, spec in
`docs/control_function.md`) scores the state for the acting party:

- `g`: the normalized margin between the opponent's offer and the party's personality-adjusted reservation price
- `z`: how far the party's own offer sits outside its acceptable range
- `A_k = (e^{αt} − 1)/(e^α − 1)`: round pressure, rising from 0 to 1 over the turn budget
- `s = 1 − V/V_max`, where `V = w_z·z + w_g·g`: a completion score

These become action values:

```
q_price_adjust = h_z (c_g + c_z)
q_extra        = c_x · A_k^γ
q_close        = h_c · s + h_c2 · [s stalled and s > s_c]
q_clarify      = h_l (1 − s)
```

A temperature softmax turns them into a distribution (temperature 1.5 while `s < s_c`, then
0.75), and the action is **sampled** from it, not taken greedily, so dialog flow varies across
runs. Some actions force the partner's next move: a close (or an ultimatum, available only to
high-`eta` personalities under high round pressure) must be answered with accept or reject,
decided by the partner's `p`, and a clarification request must be answered.

Given the action, the **prompt manager** chooses among that action's prompts by cosine
similarity between each prompt's hand-assigned 15-trait vector and the speaker's personality
vector. It never reuses a prompt within a negotiation. **Prices are computed, not
generated.** The buyer opens at `alpha · seller_min`, and later concessions follow a Faratin
model, `P_k = P_other + (P_own − P_other)·e^{−c·k}`, applied with probability `p`. The chosen
prompt, with the price filled in and the running dialog, goes to Qwen2.5-14B-Instruct (4-bit),
which writes a single utterance zero-shot.

I used zero-shot because small models in my experience tended to degrade very quickly when you let
a chat instance accumulate its own context and didnt manage it externally. Since it was one model generating 
the whole conversation, zero-shot gave it more variation. Small models don't have much room for variance
so if the negotiation was cached as it went it would push the model into a rut of generating many very
similar samples. I chose to use a small local model because calling an API for an LLM to generate the 
same data was expensive for not much gain in quality.

Completed runs pass through `utils/sft_converter.py`, which emits each negotiation twice, once
with each side as the assistant, tagged with personality, difficulty and outcome. They are then
cleaned and analyzed with `scripts/utils/` (unicode and emoji cleanup, third-person filtering,
duplicate-seed checks, difficulty bins, token counts). `scripts/strategy_labeler.py` uses Claude
to tag each packet with its key negotiation strategy. `convodata/` imports general
conversational data (OASST1, DeepDialogue), which the curriculum mixes in.

The negotiation dataset used for training held **10,770** SFT conversations, alongside
**10,000** general conversational examples. Related work that informed the design is
summarized in `docs/pact_info.md`.

## Training (`training-lab/`)

**Model and adapters.** Mistral-7B-v0.1 in 4-bit NF4 with double quantization and bf16
compute. LoRA with r=32, α=64 and dropout 0.05 on all attention and MLP projections.
`paged_adamw_8bit`, cosine schedule, effective batch 16, 1,280-token sequences, Flash
Attention 2 (`training_parameters.py`).

**Next-utterance format.** `data_processing/reformat_to_next_utterance.py` splits each
conversation into 2–3 prefixes (early, middle, final), each ending on an assistant turn. It
records character-level `masking_regions` for system, user and assistant spans. Only the final
assistant turn of each prefix is the training target. Earlier assistant turns count as context.

**Weighted masking.** The first models trained with standard binary masking (assistant tokens
on, everything else -100) learned to write *both* sides of the conversation. Instead of
removing context from the loss, `training/curriculum/masking_collator.py` assigns per-token
weights, and `training/weighted_trainer.py` computes the loss normalized by the sum of the
weights. Character regions map to tokens through the tokenizer's offset mapping, re-encoded
exactly as the ids were built (BOS included). The collator refuses to train if the ids don't
match.

**Curriculum** (`curriculum_parameters.py`, guide in `docs/CURRICULUM_GUIDE.md`):

| Phase | Conversational / negotiation mix | Context (user) weight | LR multiplier |
|---|---|---|---|
| 1. contextual learning | 50 / 50 | 0.05 | 0.7 |
| 2. guided practice | 20 / 80 | 0.02 | 1.0 |
| 3. specialized training | 5 / 95 | 0.0 | 0.6 |

These are the default phase definitions. The file also has 500- and 3,000-step presets, and
the 3,000-step validation preset is the one currently active. Mixing in general conversation
guards against catastrophic forgetting while the model
specializes. A **health monitor** (`training/curriculum/monitor.py`) generates from fixed
validation prompts during training and checks for repetition, length collapse and token
collapse. On collapse it triggers a recovery protocol: LR drops to 10%, the mix shifts to 70%
conversational, and masking is disabled for 200 steps. Phase checkpoints allow resuming from
any phase with adjusted parameters (`resume_curriculum.py`).

**Evaluation.** `evaluation/run_selfchat.py` has the adapter play both buyer and seller on
held-out seeds and flags degenerate turns. `evaluation/run_interactive.py` lets a
human play the other side.

## Running it

Both projects target Python 3.10 and a CUDA GPU (4-bit bitsandbytes, optional flash-attn).
Commands run from inside each project directory.

```bash
# --- dataset-creator ---
cd dataset-creator && pip install -r requirements.txt
python scripts/simulation_runner.py --seed-file seeds/examples/tech_seeds.json --num-negotiations 10
python utils/sft_converter.py data/runs/<run_dir> -o data/sft/negotiation.jsonl --seed-dir seeds/examples

# --- training-lab ---
cd ../training-lab && pip install -r requirements.txt
python -m training.converters.messages_to_sft ../dataset-creator/data/sft/negotiation.jsonl \
    -o data/training_data/negotiation-dataset_formatted.jsonl
python data_processing/reformat_to_next_utterance.py data/training_data/negotiation-dataset_formatted.jsonl
python -m data_processing.pre_tokenize_datasets \
    --conversational-data data/training_data/conversational_next_utterance.jsonl \
    --negotiation-data data/training_data/negotiation-dataset_formatted_next_utterance.jsonl
python train_curriculum.py \
    --conversational-data data/training_data/conversational_next_utterance_tokenized.jsonl \
    --negotiation-data data/training_data/negotiation-dataset_formatted_next_utterance_tokenized.jsonl \
    --output-dir models/curriculum
python -m evaluation.run_selfchat --adapter-path models/curriculum \
    --seed-file ../dataset-creator/seeds/examples/tech_seeds.json --num-seeds 5 --verbose
```

Tests run on CPU (the masking tests download the Mistral tokenizer):

```bash
cd training-lab && pytest tests/
cd dataset-creator && pytest tests/
```

## Known limitations

- Training examples end at the target turn without an EOS token, so the model never learns to
  stop. Evaluation relies on stop sequences (`User:`, `Assistant:`, blank line).
- Accept/reject after a close depends only on the partner's personality `p`, not on whether the
  price falls inside its range. The data teaches negotiation style and tactics more than
  price-optimal play.
- The generated dataset itself is not included.

## Layout

```
dataset-creator/
  scripts/simulation_runner.py   negotiation simulator (entry point)
  utils/control_function.py      action-selection math
  utils/price_dynamics.py        opening offer + Faratin concessions
  utils/prompt_manager.py        trait-similarity prompt selection
  prompts/                       personality deck + per-action prompt pools with trait vectors
  utils/sft_converter.py         runs → SFT packets
  scripts/utils/                 dataset cleaning and analysis
  convodata/                     general conversational data importers
  docs/                          control function, simulation flow, prompt design
training-lab/
  train_curriculum.py            training entry point (resume_curriculum.py to resume)
  curriculum_parameters.py       phases, data mix, loss weights
  training_parameters.py         QLoRA / trainer hyperparameters
  training/curriculum/           wrapper, sampler, masking collator, monitor, checkpoints
  training/converters/           messages → training text
  data_processing/               next-utterance reformatting, pre-tokenization, analysis
  evaluation/                    self-chat and interactive evaluation
```

## License

MIT
