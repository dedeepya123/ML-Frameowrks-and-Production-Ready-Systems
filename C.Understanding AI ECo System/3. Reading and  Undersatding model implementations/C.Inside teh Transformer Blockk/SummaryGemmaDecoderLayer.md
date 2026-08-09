Think of DecoderLayer as a container that says:

"Take the current hidden representation, let Attention update it, then let the MLP update it, and pass the result to the next layer."

1. The complete picture

A simplified pre-norm decoder layer:

                 hidden_states
                       │
                       ├───────────────┐
                       │               │
                       ↓               │
                    Norm              │
                       ↓               │
                   Attention           │
                       ↓               │
                       └────── + ──────┘
                              │
                         hidden_states
                              │
                              ├───────────────┐
                              │               │
                              ↓               │
                            Norm             │
                              ↓               │
                            MLP              │
                              ↓               │
                              └────── + ──────┘
                                      │
                                      ↓
                                  output

That's the whole decoder layer.

2. A realistic small implementation
class DecoderLayer(nn.Module):

    def __init__(self, config):
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = GatedMLP(config)

    def forward(
        self,
        hidden_states,
        position_ids=None,
        attention_mask=None,
    ):

        # Attention block
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(
            hidden_states
        )

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states

Don't memorize this code. Let's identify the objects.

3. __init__ tells you the architecture

Look at:

self.input_layernorm
self.self_attn
self.post_attention_layernorm
self.mlp

Immediately translate:

DecoderLayer
│
├── normalization
├── attention
├── normalization
└── MLP

This is one of the most useful things to do when opening an unfamiliar model.

The constructor often reveals the architecture before you even read forward().

4. forward() reveals the execution order

First:

residual = hidden_states

We're preserving the residual stream.

Then:

hidden_states = self.input_layernorm(hidden_states)

Normalize.

Then:

hidden_states = self.self_attn(...)

Attention performs:

Q
K
V
↓
RoPE
↓
QKᵀ
↓
mask
↓
softmax
↓
weighted V
↓
output projection

Then:

hidden_states = residual + hidden_states

Attention update gets added back to the residual stream.

5. Then the MLP block

We repeat the same pattern:

residual = hidden_states

Save current residual stream.

Then:

hidden_states = self.post_attention_layernorm(
    hidden_states
)

Normalize.

Then:

hidden_states = self.mlp(hidden_states)

which internally does:

gate_proj
      ↓
    SiLU
      ↓
      ×
      ↑
  up_proj
      ↓
 down_proj

Then:

hidden_states = residual + hidden_states

Second residual update.

Finally:

return hidden_states
6. The shape never becomes mysterious

If:

hidden_states = [B,S,H]

then throughout the decoder layer:

Input
[B,S,H]
   ↓
Norm
[B,S,H]
   ↓
Attention
[B,S,H]
   ↓
Residual
[B,S,H]
   ↓
Norm
[B,S,H]
   ↓
MLP
[B,S,H]
   ↓
Residual
[B,S,H]
   ↓
Output
[B,S,H]

Internally Attention and MLP temporarily use different shapes, but the interface of the DecoderLayer remains [B,S,H] → [B,S,H].

This is a very useful abstraction.

7. This is how you should read a real DecoderLayer

When you encounter something like:

class GemmaDecoderLayer(...)

don't immediately read every line.

First identify:

1. What normalization?
2. What attention class?
3. What MLP class?
4. Where are residuals added?
5. What enters and exits the layer?

Then create:

              DecoderLayer
                   │
          ┌────────┴────────┐
          ↓                 ↓
       Attention            MLP
          │                 │
     Norm → Attn         Norm → MLP
          │                 │
       Residual          Residual

Then only investigate deviations.

For example, if the model has:

post_attention_layernorm

you already know what it likely means.

If you see:

pre_feedforward_layernorm

same reasoning.

If you see something unusual:

some_extra_scale(...)

that's when you stop and investigate.

8. One important code-reading principle

You are going to encounter implementations that look very different:

x = x + self.attn(self.norm(x))
x = x + self.mlp(self.norm2(x))

or:

residual = x
x = self.norm(x)
x = self.attn(x)
x = x + residual

residual = x
x = self.norm2(x)
x = self.mlp(x)
x = x + residual

or heavily optimized code where operations are fused.

Don't get distracted by syntax.

Translate them into the same conceptual graph:

Norm → Attention → Residual
Norm → MLP       → Residual

Data flow is more important than code formatting.

9. Where we are now

We can now recognize the major classes:

Model
 │
 ├── Embedding
 │
 ├── DecoderLayer × N
 │       │
 │       ├── Norm
 │       ├── Attention
 │       │     ├── Q/K/V
 │       │     ├── Heads
 │       │     ├── RoPE
 │       │     ├── Mask
 │       │     └── Attention computation
 │       │
 │       ├── Residual
 │       │
 │       ├── Norm
 │       ├── Gated MLP
 │       │     ├── gate_proj
 │       │     ├── up_proj
 │       │     └── down_proj
 │       │
 │       └── Residual
 │
 └── output / LM head

That is a very strong architecture-level map for reading decoder-only LLM code.
