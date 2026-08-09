# Module 4 — Inside the Transformer Block

## Lesson 10 — Softmax → Attention Weights → Weighted Sum of V

## 1. Attention scores

After Q/K projections, head splitting, RoPE, and scaling:

    QKᵀ / sqrt(head_dim)

produces attention scores.

Shape:

    [B, H, S, S]

The final two dimensions mean:

    query positions × key positions

A score represents how relevant a key is to a query.

Scores are not probabilities.

---

## 2. Causal masking

Before softmax, future-token scores are masked.

Example:

    [2.1, -∞, -∞]

This prevents the query from attending to future tokens.

---

## 3. Softmax

Softmax converts each query's scores over the key dimension into a probability distribution.

Conceptually:

    scores
      ↓
    softmax
      ↓
    attention weights

Code commonly looks like:

    torch.softmax(attn_scores, dim=-1)

`dim=-1` is the key-position dimension.

Therefore, for every:

    batch
    head
    query position

the weights across all key positions sum to 1.

---

## 4. Attention weights

Example:

    [0.2, 0.5, 0.3]

means:

    20% information from V0
    50% information from V1
    30% information from V2

The weights determine how much information to retrieve from each value vector.

---

## 5. Weighted sum of V

The attention output is:

    softmax(QKᵀ / sqrt(D) + mask) V

For one query:

    weights = [0.2, 0.5, 0.3]

and:

    V0 = [1,2]
    V1 = [4,6]
    V2 = [10,20]

then:

    output
    = 0.2V0 + 0.5V1 + 0.3V2
    = [5.2, 9.4]

So attention weights select and combine information from V.

---

## 6. Shape transformation

Q:

    [B,H,S,D]

K:

    [B,H,S,D]

V:

    [B,H,S,D]

QKᵀ:

    [B,H,S,S]

softmax:

    [B,H,S,S]

attention_weights @ V:

    [B,H,S,D]

The key dimension `S` is transformed back into the value dimension `D`.

---

## 7. Meaning of Q, K, V

Q:

    "What information am I looking for?"

K:

    "How relevant am I to a query?"

V:

    "What information do I provide if selected?"

QKᵀ determines relevance.

Softmax converts relevance scores into attention weights.

Weights × V retrieves the information.

---

## 8. Multi-head attention

Each head independently performs:

    Q_h K_hᵀ
        ↓
      softmax
        ↓
     weights
        ↓
     weights V_h
        ↓
    head output

If:

    Q.shape = [B,H,S,D]

then:

    attention output = [B,H,S,D]

Each head can learn different attention relationships.

---

## 9. Combining heads

After attention:

    [B,H,S,D]

is rearranged into:

    [B,S,H,D]

and reshaped into:

    [B,S,H×D]

Since:

    H × D = hidden_size

the combined representation returns to:

    [B,S,hidden_size]

---

## 10. Output projection

The concatenated head representation is passed through:

    o_proj

Conceptually:

    per-head outputs
        ↓
    concatenate
        ↓
    o_proj
        ↓
    attention output

`o_proj` allows the model to learn how information from the different heads should be mixed.

---

## 11. Complete self-attention pipeline

    hidden_states
         ↓
    q_proj / k_proj / v_proj
         ↓
       Q / K / V
         ↓
      split heads
         ↓
       RoPE(Q,K)
         ↓
       Q' / K'
         ↓
        Q'K'ᵀ
         ↓
      scale by sqrt(D)
         ↓
      causal mask
         ↓
       softmax
         ↓
    attention weights
         ↓
       weights × V
         ↓
    per-head outputs
         ↓
    concatenate heads
         ↓
       o_proj
         ↓
    attention output

---

## 12. Two conceptual stages

### Determine where to look

    Q
    K
    ↓
    QKᵀ
    ↓
    softmax
    ↓
    attention weights

### Retrieve information

    attention weights
           ×
           V
           ↓
    attention output

---

## 13. Code-reading checklist

When reading an unfamiliar attention class, identify:

1. Where Q/K/V are created
2. Their shapes
3. Where heads are created
4. Where RoPE is applied
5. Where QKᵀ is calculated
6. Where scaling occurs
7. Where causality is enforced
8. Where softmax occurs
9. Where attention weights multiply V
10. Where heads are combined
11. Where `o_proj` is applied

If these can be identified, the overall attention implementation can usually be understood even when the code structure differs between models.
``` text
                  hidden_states
                       │
              ┌────────┼────────┐
              ↓        ↓        ↓
           q_proj    k_proj    v_proj
              ↓        ↓        ↓
              Q        K        V
              │        │
              ↓        ↓
          split into heads
              │        │
              ↓        ↓
             RoPE     RoPE
              │        │
              ↓        ↓
             Q'       K'
              │        │
              └────┬───┘
                   ↓
                 Q'K'ᵀ
                   ↓
              / √head_dim
                   ↓
              causal mask
                   ↓
                softmax
                   ↓
           attention weights
                   │
                   ×
                   │
                   V
                   ↓
           per-head output
                   ↓
            concatenate heads
                   ↓
                 o_proj
                   ↓
          attention output
```
<img width="1187" height="640" alt="image" src="https://github.com/user-attachments/assets/1aa5debd-32d0-4d0a-97b2-c58c1b3e7e34" />
<img width="807" height="646" alt="image" src="https://github.com/user-attachments/assets/0ea3e4d2-9a77-4e81-8a24-4c7a51138649" />
<img width="1142" height="510" alt="image" src="https://github.com/user-attachments/assets/881235bb-25f9-48d4-9596-f2a3f4236796" />
<img width="1085" height="557" alt="image" src="https://github.com/user-attachments/assets/d8cb506c-7a0f-436e-8e38-6551e67c2726" />
<img width="1201" height="485" alt="image" src="https://github.com/user-attachments/assets/86b35606-e23a-453e-845c-52d18b474317" />

Ask these questions in order:

1. Where are Q/K/V created?
q_proj
k_proj
v_proj
2. What are their shapes?
[B,S,H×D]

or:

[B,H,S,D]
3. Where are heads created?

Look for:

view
reshape
transpose
permute
4. Where is positional information applied?

Look for:

rotary
rope
cos
sin
position_ids
5. Where are Q/K compared?

Look for:

matmul
Q @ Kᵀ
6. Where is scaling?

Look for:

sqrt(head_dim)
7. Where is causality enforced?

Look for:

mask
causal
is_causal
8. Where does softmax happen?

Usually:

dim=-1
9. Where are values retrieved?

Look for:

attention_weights @ V
10. Where are heads combined?

Look for:

transpose
reshape
view
11. Where is the final projection?

Look for:

o_proj

If you can answer those 11 questions, you can understand the structure of almost any standard multi-head attention implementation, even if the code is written differently.




