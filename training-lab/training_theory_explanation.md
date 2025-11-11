# Training Duration and Pattern Learning Theory

## The Two-Phase Learning Problem

### Phase 1: Surface Pattern Memorization (Steps 0-2000)
During early training with high learning rates, models first learn:
- High-frequency token patterns
- Position-based predictions
- Simple n-gram associations
- Shortcuts and statistical regularities

**This is where your model got stuck!** It learned that `eller` frequently appears at certain positions but never progressed to understanding *why*.

### Phase 2: Deep Pattern Integration (Steps 2000-8000+)
With continued training at appropriate learning rates:
- Surface patterns get refined by context
- Model learns conditional dependencies
- Style and behavior patterns emerge from statistics
- Shortcuts get replaced by actual understanding

## Why Short Training Causes Token Collapse

```
Training Steps:   0 -------- 1000 -------- 2000 -------- 3000 -------- 4000+
                  ^            ^             ^             ^             ^
Model State:      Memorizing   Overfitting   Should be     Contextual   Deep
                  Fragments    to Shortcuts  Generalizing  Integration  Understanding

Your Training:    [============STOPPED HERE==]
                                              ^
                                    Model collapsed into
                                    memorized fragments
```

### The Critical Insight: Negotiation Style Training

For behavior-heavy training like negotiation:

1. **Token Prediction != Behavior Understanding**
   - Steps 0-1000: Model learns "seller says price words"
   - Steps 1000-3000: Model learns "seller defends price when buyer objects"
   - Steps 3000-5000: Model learns negotiation *strategy* patterns
   - Steps 5000+: Model integrates style with coherent generation

2. **High Learning Rate + Short Training = Catastrophe**
   - LR 5e-5 causes rapid memorization
   - Stopping at 3000 steps freezes model in "shortcut" phase
   - Result: Spams memorized fragments like "eller", "0000", "40"

## Masking Strategy: Why Full Visibility First Matters

### Traditional Approach (What You Did):
```
MASKED    VISIBLE   MASKED    VISIBLE   MASKED
[System]→[Assistant]→[User]→[Assistant]→[User]
```
**Problem:** Model never sees full conversation flow, learns isolated response generation

### Recommended Approach:
```
Phase 1 (Contextual Learning):
MASKED    VISIBLE   VISIBLE   VISIBLE   VISIBLE
[System]→[Assistant]→[User]→[Assistant]→[User]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         Model sees full exchange patterns

Phase 2 (Specialized Training):
MASKED    VISIBLE   MASKED    VISIBLE   MASKED
[System]→[Assistant]→[User]→[Assistant]→[User]
         Focus on assistant-only generation
```

## Answering Your Specific Questions

### 1. "Should system tags stay masked?"
**Yes, always mask system prompts.** They're meta-instructions, not conversation content. Exposing them causes the model to generate system-like instructions in responses.

### 2. "Does the model need to learn negotiation before responses?"
**Absolutely yes!** Think of it as a curriculum:
- **Grade 1:** Learn what negotiation looks like (full visibility)
- **Grade 2:** Learn to participate in negotiation (partial masking)
- **Grade 3:** Master negotiation responses (full masking)

Your approach jumped straight to Grade 3, causing the model to mimic surface patterns without understanding.

### 3. "Is this why it hyperfixates on one-sided tokens?"
**Exactly!** With only assistant turns visible:
- Model can't learn cause-effect (User says X → Assistant responds Y)
- Can't understand conversational flow
- Learns position-based token statistics instead of semantic responses
- Develops "token tunnel vision" - predicting next token without conversation awareness

## The "eller" Token Bleeding Explained

Here's what likely happened:
1. Model saw Assistant responses after [MASKED] regions
2. Learned statistical pattern: "After position X, 'eller' token appears"
3. Without seeing "Seller:" in context, couldn't learn it's wrong
4. High LR + short training = pattern crystallized before correction

This is **phantom token learning** - the model learns tokens it never directly saw through:
- Positional encoding patterns
- Attention weight spillover
- Statistical co-occurrence with visible tokens

## Optimal Training Strategy for Your Case

```python
# training_phases.py

def phase_1_contextual(steps=2000):
    """Learn negotiation structure"""
    return {
        "masking": "system_only",  # See full conversations
        "lr": 5e-6,  # Very gentle
        "data_mix": "50% conversational, 50% negotiation"
    }

def phase_2_guided(steps=2000):
    """Learn role-specific behavior"""
    return {
        "masking": "user_assistant_alternating",
        "lr": 1e-5,  # Increase slightly
        "data_mix": "30% conversational, 70% negotiation"
    }

def phase_3_specialized(steps=4000):
    """Master assistant responses"""
    return {
        "masking": "assistant_only_visible",
        "lr": 8e-6,  # Reduce for fine-tuning
        "data_mix": "10% conversational, 90% negotiation"
    }
```

## Key Takeaway

Your model didn't have time to learn negotiation - it only had time to memorize token fragments. The "eller" spam is a symptom of **statistical memorization without semantic understanding**. Longer training with progressive masking allows:

1. **Context understanding** before response generation
2. **Pattern abstraction** beyond token memorization
3. **Style integration** with maintained coherence

Think of it like teaching someone to dance by only showing them their own feet, never their partner's movements or the music. They'll develop weird, repetitive movements that don't match the actual dance!