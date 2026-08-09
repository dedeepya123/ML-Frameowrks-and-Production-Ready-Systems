# Reading Transformer Model Code — Quick Reference

## 1. Start at the Framework Level

When using Hugging Face:

```python
AutoModelForCausalLM.from_pretrained(...)
```

the framework does roughly:

```text
checkpoint/config
      ↓
model_type
      ↓
registry / dispatch
      ↓
model-specific class
      ↓
construct model
      ↓
load checkpoint weights
```

**Key idea:** `AutoModel` is a dispatcher/factory. It does not itself contain the model architecture.

---

## 2. `model_type` → Architecture

The config tells the framework which architecture to instantiate.

Example:

```text
model_type = "gemma"
        ↓
Gemma architecture
```

When exploring an unfamiliar repository:

```text
config.py
   ↓
model_type
   ↓
find corresponding model class
```

---

## 3. Constructor vs Checkpoint Loading

These are separate steps.

### Constructor

Builds the model structure:

```text
Model
 ├── Embedding
 ├── DecoderLayer × N
 │    ├── Attention
 │    ├── MLP
 │    └── Norm
 └── Final Norm
```

and creates parameter objects with expected shapes.

### Checkpoint loading

Loads the stored tensor values into those parameters.

Conceptually:

```text
checkpoint tensor
      ↓
parameter name
      ↓
matching model parameter
```

So:

> **Constructor creates the structure; checkpoint loading populates the structure with learned values.**

---

## 4. Python Inheritance

If:

```python
class Dog(Animal):
    ...
```

and:

```python
d = Dog()
```

then:

```python
type(d) is Dog
```

but:

```python
isinstance(d, Dog)     # True
isinstance(d, Animal)  # True
```

The object's **actual type is the child class**, but it is also an instance of its parent classes.

---

# Model Class Hierarchy

Typical structure:

```text
PreTrainedModel
      ↑
GemmaPreTrainedModel
      ↑
GemmaForCausalLM
```

and separately:

```text
GemmaModel
      ↑
GemmaPreTrainedModel
```

Conceptually:

```text
GemmaForCausalLM
      ↓
GemmaModel
      ↓
DecoderLayer × N
      ↓
Attention + MLP + Norm
```

When a method isn't defined in the child, Python searches the inheritance hierarchy for it.

---

# `__init__()` Reading

The constructor tells us:

> **What objects make up this model?**

Look for:

```python
self.embed_tokens
self.layers
self.norm
self.self_attn
self.mlp
```

Build the architecture tree before reading the computation.

Example:

```text
Model
 ├── Embedding
 ├── Layers
 │    ├── Attention
 │    ├── MLP
 │    └── Norm
 └── Final Norm
```

---

# `forward()` Reading

The constructor tells us **what exists**.

The `forward()` tells us:

> **How data flows through those objects.**

Always ask:

```text
What enters?
What leaves?
What is the shape?
What operation happens?
Why?
Who consumes the output?
```

---

# Core Tensor Flow

For a decoder-only LLM:

```text
input_ids
    ↓
Embedding
    ↓
hidden_states [B,S,H]
    ↓
DecoderLayer × N
    ↓
Final Norm
    ↓
hidden_states
    ↓
LM Head
    ↓
logits [B,S,V]
```

Where:

```text
B = batch size
S = sequence length
H = hidden size
V = vocabulary size
```

---

# DecoderLayer

Think of one layer as:

```text
hidden_states
      ↓
    Norm
      ↓
  Attention
      ↓
  Residual Add
      ↓
    Norm
      ↓
     MLP
      ↓
  Residual Add
      ↓
hidden_states
```

Interface:

```text
[B,S,H] → [B,S,H]
```

---

# Attention

High-level flow:

```text
hidden_states
      ↓
   Q / K / V
      ↓
 split into heads
      ↓
     RoPE
      ↓
    QKᵀ
      ↓
 causal mask
      ↓
   softmax
      ↓
 attention × V
      ↓
 merge heads
      ↓
 output projection
      ↓
[B,S,H]
```

### Important distinction

Attention:

> **mixes information across tokens.**

---

# RoPE

RoPE applies position-dependent rotations to Q and K.

Mental model:

```text
position
   +
frequency
   ↓
angle
   ↓
sin / cos
   ↓
rotate Q and K
```

Key idea:

> RoPE injects positional information into Q/K without adding a separate positional embedding to the hidden states.

---

# Causal Mask

For autoregressive generation:

```text
Token i
  ↓
can attend to:
  current + previous tokens

cannot attend to:
  future tokens
```

Conceptually:

```text
      K0 K1 K2 K3
Q0    ✓  ✗  ✗  ✗
Q1    ✓  ✓  ✗  ✗
Q2    ✓  ✓  ✓  ✗
Q3    ✓  ✓  ✓  ✓
```

During cached inference, the implementation may not need to materialize the full training-style mask because only valid past/current K/V are available.

---

# Normalization

For:

```text
x = [B,S,H]
```

normalization generally operates over:

```text
H
```

independently for each token.

### LayerNorm

```text
subtract mean
      ↓
divide by standard deviation
      ↓
learned scale + bias
```

### RMSNorm

```text
square
  ↓
mean
  ↓
sqrt
  ↓
divide
  ↓
learned scale
```

Modern LLMs commonly use RMSNorm.

Recognize RMSNorm code through patterns such as:

```python
x.pow(2)
mean(dim=-1)
rsqrt(...)
x * weight
```

---

# MLP / Gated MLP

Typical gated MLP:

```text
x
├──→ gate_proj → activation ──┐
│                             ×
└──→ up_proj ─────────────────┘
                 ↓
             down_proj
                 ↓
              output
```

Purpose:

> **Transform each token's representation independently.**

Unlike attention, MLP does not mix sequence positions.

---

# Attention vs MLP vs Normalization

```text
Attention
→ mixes tokens

MLP
→ transforms each token

Normalization
→ normalizes features within each token
```

This distinction is extremely useful when reading unfamiliar code.

---

# Model-Level Composition

The main model repeatedly applies the same DecoderLayer:

```python
for layer in self.layers:
    hidden_states = layer(hidden_states)
```

Think:

```text
Embedding
   ↓
Layer 0
   ↓
Layer 1
   ↓
Layer 2
   ↓
...
   ↓
Layer N
   ↓
Final Norm
```

`nn.ModuleList` registers the layers with PyTorch so their parameters/state are tracked correctly.

---

# `GemmaModel` vs `GemmaForCausalLM`

### `GemmaModel`

Produces hidden representations:

```text
input
 ↓
GemmaModel
 ↓
hidden_states
```

### `GemmaForCausalLM`

Adds the language-model head:

```text
input
 ↓
GemmaModel
 ↓
hidden_states
 ↓
LM Head
 ↓
logits
```

So:

> **Model = representation network**

> **ForCausalLM = representation network + vocabulary prediction head**

---

# Production Code Reading Strategy

Don't read a large model repository linearly.

Use:

```text
config.py
   ↓
model_type
   ↓
architecture class
   ↓
__init__()
   ↓
class hierarchy
   ↓
forward()
   ↓
trace hidden_states
   ↓
inspect components
   ↓
verify tensor shapes
   ↓
identify unusual/optimized code
```

---

# The 8 Questions to Ask

Whenever you encounter unfamiliar model code:

```text
1. What class am I in?

2. Where does this class sit in the hierarchy?

3. What is this class responsible for?

4. What is the input?

5. What is the output?

6. What are the tensor shapes?

7. Why is this operation here?

8. Who consumes this output next?
```

If you can answer these, you are **understanding the code rather than memorizing it.**

---

# The Core Mental Model

When reading any decoder-only Transformer:

```text
Framework
   ↓
Architecture selection
   ↓
Model class
   ↓
Embedding
   ↓
DecoderLayer × N
   ├── Norm
   ├── Attention
   │    ├── Q/K/V
   │    ├── Heads
   │    ├── RoPE
   │    └── Mask
   ├── Residual
   ├── Norm
   ├── MLP
   └── Residual
   ↓
Final Norm
   ↓
LM Head
   ↓
Logits
```

The goal is not to memorize this diagram.

The goal is to be able to **reconstruct it yourself when you open an unfamiliar model repository.**
