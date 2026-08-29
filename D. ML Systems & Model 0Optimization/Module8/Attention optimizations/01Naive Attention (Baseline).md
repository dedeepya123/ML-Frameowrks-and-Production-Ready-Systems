# Module 7 — Lesson 1: Naive Attention (Baseline)

## Attention Pipeline

Q, K, V
   ↓
QKᵀ
   ↓
Softmax
   ↓
Scores × V
   ↓
Output

## Tensor Shapes

Q: [N, d]
K: [N, d]
V: [N, d]

Scores: [N, N]

The score matrix grows quadratically with sequence length.

## Runtime View

Kernel 1:
Read Q, K
Write Scores

Kernel 2:
Read Scores
Write Scores

Kernel 3:
Read Scores, V
Write Output

## Biggest Intermediate

The largest temporary tensor is the score matrix.

It is repeatedly written and read during execution.

## Compute vs Memory

QKᵀ:
- heavy computation
- writes Scores

Softmax:
- reads/writes Scores

Scores×V:
- heavy computation
- reads Scores again

## Five Perspectives

- Model: attention computation
- Tensor: tensors created
- Compute: matrix multiplications
- Memory: tensor movement
- Runtime: kernels + memory placement

## Core Insight

Naive attention's biggest systems issue isn't just computation.
It repeatedly materializes and moves a large `Scores` tensor, which becomes increasingly expensive as sequence length grows.
