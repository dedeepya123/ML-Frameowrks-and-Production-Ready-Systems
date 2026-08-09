# Module 6 — Model-Level Composition

## Lesson 1 — Main Model and Decoder Layer Stack

## 1. High-level model structure

A decoder-only Transformer can be viewed as:

    input_ids
        ↓
    Token Embedding
        ↓
    hidden_states
        ↓
    DecoderLayer × N
        ↓
    Final Norm
        ↓
    hidden_states
        ↓
    LM Head
        ↓
    logits

The model consists of repeated DecoderLayer instances.

---

## 2. Decoder layer stack

The model typically creates:

    self.layers = nn.ModuleList([
        DecoderLayer(config)
        for _ in range(config.num_hidden_layers)
    ])

If:

    num_hidden_layers = 32

then the model contains:

    DecoderLayer 0
    DecoderLayer 1
    ...
    DecoderLayer 31

All layers follow the same architectural structure but have their own parameters.

---

## 3. Why ModuleList?

Use:

    nn.ModuleList

instead of a normal Python list because ModuleList registers the contained modules with PyTorch.

Therefore PyTorch can correctly track:

    parameters
    buffers
    state_dict entries
    device placement
    dtype conversion
    train/eval state

This connects directly to the PyTorch module hierarchy.

---

## 4. Main forward flow

Conceptually:

    hidden_states = embed_tokens(input_ids)

    for layer in self.layers:
        hidden_states = layer(hidden_states)

    hidden_states = final_norm(hidden_states)

    return hidden_states

The main Model does not need to know the internal implementation of Attention or MLP.

It composes DecoderLayer objects.

---

## 5. hidden_states

`hidden_states` is the current learned representation of the sequence.

Conceptually:

    input_ids
        ↓
    Embedding
        ↓
    H⁰
        ↓
    Layer 0
        ↓
    H¹
        ↓
    Layer 1
        ↓
    H²
        ↓
    ...
        ↓
    Hᴺ

Each DecoderLayer updates the representation.

---

## 6. Shape interface

If:

    input_ids = [B,S]

and:

    hidden_size = H

then:

    Embedding:
        [B,S] → [B,S,H]

Each DecoderLayer:

    [B,S,H] → [B,S,H]

Final Norm:

    [B,S,H] → [B,S,H]

Therefore the model's hidden representation keeps the same external shape throughout the decoder stack.

---

## 7. DecoderLayer abstraction

The Model sees:

    DecoderLayer:
        input  → [B,S,H]
        output → [B,S,H]

It does not need to know that internally the layer contains:

    Norm
    Attention
    Residual
    Norm
    MLP
    Residual

This is abstraction through composition.

---

## 8. Nested model hierarchy

Think of the architecture as nested composition:

    Model
    │
    ├── Embedding
    │
    ├── DecoderLayer
    │     ├── Attention
    │     │     ├── Q/K/V
    │     │     ├── RoPE
    │     │     ├── Mask
    │     │     └── attention computation
    │     │
    │     └── MLP
    │           ├── gate
    │           ├── up
    │           └── down
    │
    ├── DecoderLayer
    │
    ├── ...
    │
    └── Final Norm

This allows code to be understood at different abstraction levels.

---

## 9. input_ids vs hidden_states

input_ids:

    integer token IDs

Example:

    [1542, 827, 91, 2048]

They index the vocabulary.

Embedding converts them into floating-point representations:

    input_ids
        ↓
    embedding
        ↓
    hidden_states

After embedding, the Transformer mostly operates on hidden_states.

---

## 10. Position IDs

Tokens have positions:

    token A → 0
    token B → 1
    token C → 2
    token D → 3

Conceptually:

    position_ids = [0,1,2,3]

Position IDs are used by Attention, particularly for RoPE.

The main model may prepare position information and pass it through each DecoderLayer.

Conceptually:

    hidden_states ─────────→ DecoderLayer
    position_ids ──────────→ DecoderLayer
    attention_mask ────────→ DecoderLayer

---

## 11. Training vs inference

Training:

    whole sequence can be processed together

Example:

    A B C D E

positions:

    0 1 2 3 4

Inference:

    autoregressive generation produces one/few new tokens at a time.

Previous K/V states can be stored in a KV cache.

Conceptually:

    previous tokens
        ↓
    cached K/V

    new token
        ↓
    current Q/K/V
        ↓
    Attention using current + cached K/V

The cache is part of efficient autoregressive execution.

---

## 12. Model vs ForCausalLM

A base Model typically produces hidden representations:

    GemmaModel
        ↓
    hidden_states

A causal language-model wrapper adds the language-model head:

    GemmaForCausalLM
        ↓
    GemmaModel
        ↓
    Final hidden_states
        ↓
    LM Head
        ↓
    logits

Therefore:

    Model = representation-producing network

    ForCausalLM = representation network + next-token prediction head

---

## 13. LM Head

Suppose:

    hidden_size = 2048
    vocab_size = 256000

Then:

    [B,S,2048]
        ↓
    LM Head
        ↓
    [B,S,256000]

For each sequence position, the model produces a logit for every vocabulary token.

Conceptually:

    hidden representation
        ↓
    vocabulary scores
        ↓
    next-token selection

---

## 14. Code-reading strategy

When opening a new model repository:

1. Read config.py.
2. Identify:
       model_type
       hidden_size
       num_hidden_layers
       num_attention_heads
       num_key_value_heads
       intermediate_size
       vocab_size

3. Find the architecture classes.

4. Build the hierarchy:

       ForCausalLM
           ↓
       Model
           ↓
       DecoderLayer
           ↓
       Attention
           ↓
       MLP

5. Read the Model forward.

6. Track `hidden_states`.

7. Identify what additional state is passed:
       position_ids
       attention_mask
       cache

8. Then zoom into individual classes only when needed.

---

## 15. Core mental model

When seeing:

    for layer in self.layers:
        hidden_states = layer(
            hidden_states,
            ...
        )

think:

    "The model is passing the current representation
     through the Transformer layer stack."

The important data flow is:

    input_ids
        ↓
    embedding
        ↓
    hidden_states
        ↓
    layer 0
        ↓
    layer 1
        ↓
    ...
        ↓
    layer N
        ↓
    final norm
        ↓
    hidden representation
        ↓
    LM head
        ↓
    logits

Architecture is understood by following:

    classes
    + data flow
    + tensor shapes
    + responsibilities
<img width="1192" height="810" alt="image" src="https://github.com/user-attachments/assets/e4a40140-327c-4996-ad62-04349345d2e2" />
    <img width="1141" height="777" alt="image" src="https://github.com/user-attachments/assets/8c7265c5-7a7b-486c-9f1d-a189b7e26305" />

  <img width="1146" height="507" alt="image" src="https://github.com/user-attachments/assets/f2353146-926e-4938-923e-35e692698387" />
