# Module 7 — Lesson 7: Production Attention Implementations

## Goal

Learn to read production attention implementations by separating:

Model
→ Mathematics
→ Algorithm
→ Framework
→ Kernel
→ Runtime
→ Hardware

---

## Start With the Mathematical Contract

Attention:

Q = XWq
K = XWk
V = XWv

S = QKᵀ / sqrt(d)

P = softmax(S)

O = PV

Everything in an optimized implementation should preserve
this mathematical function.

---

## First Check Shapes

Typical:

Q = [B,H,N,D]
K = [B,H,N,D]
V = [B,H,N,D]

Attention scores:

QKᵀ = [B,H,N,N]

N×N should immediately trigger a memory concern.

---

## Naive Reference Implementation

scores = Q @ K.T
scores = scores + mask
probs = softmax(scores)
output = probs @ V

Useful as a mathematical reference.

But scores and probs may be large materialized intermediates.

---

## What to Look For in Production Code

### 1. Mathematics

Find:

- QKᵀ
- scaling
- mask
- softmax
- PV

### 2. Fusion

Ask:

Are QKᵀ + softmax + PV separate operations?

or are they executed inside one optimized attention kernel?

Fusion can reduce intermediate memory traffic.

### 3. Tiling

Look for:

- blocks
- tiles
- chunks
- query blocks
- key blocks

Question:

Is the large attention computation being processed
in smaller pieces?

### 4. Online Softmax

Look for:

- running maximum
- running sum
- rescaling
- blockwise normalization

This can indicate online/blockwise softmax.

### 5. KV Cache

For decode, identify:

- current Q
- cached K
- cached V
- cache positions
- sequence length
- cache updates

Separate logical cache representation from physical
runtime memory management.

---

## Memory Reasoning

Ask:

Where are Q/K/V stored?

Where are intermediates stored?

How many times are they loaded?

Are they reused?

Are large intermediates written to global/off-chip memory?

A common optimized flow:

external memory
→ load tiles
→ local/on-chip memory
→ compute
→ accumulate
→ store final output

---

## Framework vs Runtime

A Python/framework expression such as:

y = q @ k.transpose(-2,-1)

does not mean Python directly executes the matrix
multiplication on the accelerator.

Conceptually:

Python/model code
→ framework operator
→ graph/operator representation
→ runtime lowering
→ optimized kernel
→ accelerator

The runtime may decide:

- kernel
- tiling
- fusion
- memory placement
- scheduling

---

## FlashAttention vs Paged Attention

FlashAttention:

Primary concern:
How to COMPUTE attention efficiently.

Key ideas:
- tiling
- fusion
- online softmax
- avoid large attention intermediates

Paged Attention:

Primary concern:
How to MANAGE/ACCESS KV cache efficiently.

Key ideas:
- KV blocks/pages
- block tables
- dynamic allocation
- memory pools
- non-contiguous physical storage

They can be combined.

---

## Production Code Reading Checklist

1. What are Q/K/V shapes?
2. Where is QKᵀ?
3. Where is scaling?
4. Where is masking?
5. Where is softmax?
6. Where is PV?
7. Is N×N materialized?
8. Is computation tiled?
9. Are operations fused?
10. Is online softmax used?
11. Where does K/V come from during decode?
12. How is KV cache represented?
13. How is KV physically allocated?
14. What data is moved between memory levels?
15. Which kernel executes the operation?
16. How does runtime select/lower the kernel?
17. Which accelerator unit performs the computation?
18. What bottleneck is the optimization removing?

---

## General Optimization Reasoning

For ANY optimization ask:

### What is expensive?

- FLOPs?
- memory capacity?
- memory bandwidth?
- communication?
- synchronization?
- allocation?
- latency?

### Where is it expensive?

- model?
- framework?
- kernel?
- memory?
- runtime?
- hardware?

### What changed?

- FLOPs?
- bytes moved?
- data reuse?
- parallelism?
- synchronization?
- memory allocation?

### What bottleneck was removed?

This is the central systems-level question.

---

## Module 7 Mental Model

Naive Attention
→ large intermediates
→ memory traffic

Memory-efficient Attention
→ avoid unnecessary materialization

FlashAttention
→ tiled + fused + online softmax execution

Tiling
→ improve locality + reuse

Paged Attention
→ efficient KV-cache memory management

Production Attention
→ combine model, algorithm, memory, runtime,
   kernel, and hardware reasoning

---

## Target Skill

When seeing an unfamiliar attention implementation:

"I will first identify the mathematical operations and
tensor shapes. Then I will determine whether the attention
matrix is materialized, whether computation is tiled/fused,
how softmax is implemented, how KV cache is accessed,
and finally how the runtime maps the computation to the
accelerator."

This is the desired ML-systems reasoning skill.
