# Module 7 — Lesson 4: FlashAttention

## What is FlashAttention?

FlashAttention is a highly optimized implementation of
memory-efficient attention.

The mathematical result remains:

O = softmax(QKᵀ)V

The main change is HOW the computation is executed.

---

## Core Problem

Naive:

QKᵀ
 ↓
materialize N×N scores
 ↓
softmax
 ↓
materialize N×N probabilities
 ↓
P×V

Large intermediate tensors cause expensive memory traffic.

---

## Core FlashAttention Idea

Process attention using tiles/blocks.

Load:

Q tile
K tile
V tile

into fast local/on-chip memory.

Then:

QKᵀ tile
    ↓
online softmax
    ↓
multiply with V tile
    ↓
update running output
    ↓
discard temporary tile

Then process the next tile.

---

## What is a Block?

A block/tile is simply a smaller chunk of a tensor.

Example:

Q = [4096 × d]

Instead of processing all 4096 queries:

Q tile = [128 × d]

Similarly:

K tile = [128 × d]
V tile = [128 × d]

Score tile:

[128 × d] × [d × 128]
        ↓
[128 × 128]

---

## Important

FlashAttention does NOT create the complete N×N
attention matrix and place it in SRAM.

It creates small score tiles, consumes them,
and discards them.

---

## Online Softmax

Full softmax normally needs the whole row.

FlashAttention maintains running state such as:

- running maximum
- running normalization term
- running output

This allows different K/V blocks to contribute to
the same final attention output.

---

## Memory Hierarchy

Conceptually:

Off-chip memory
      ↓
load Q/K/V tiles
      ↓
On-chip memory
      ↓
compute score tile
      ↓
online softmax
      ↓
accumulate output
      ↓
discard temporary tile

The exact on-chip memory structure depends on hardware.

GPU:
registers/shared memory/cache

NPU:
local SRAM/scratchpad/etc.

---

## Why It Is Faster

Main benefits:

- avoids materializing N×N intermediates
- reduces off-chip memory traffic
- improves data locality
- increases data reuse
- enables fused execution
- reduces memory footprint

It is not primarily about reducing the fundamental
O(N²d) attention computation.

---

## Fusion

Naive conceptual execution:

QKᵀ
 ↓
memory
 ↓
softmax
 ↓
memory
 ↓
P×V

FlashAttention can coordinate these operations
inside an optimized attention kernel.

---

## Roofline Connection

Arithmetic intensity:

AI = FLOPs / bytes moved

FlashAttention primarily improves the denominator:

less unnecessary off-chip memory movement

Therefore effective arithmetic intensity can increase.

---

## Hardware-Agnostic Reasoning

When analyzing an accelerator implementation,
ask:

1. What is the tile size?
2. Where are Q/K/V tiles stored?
3. Where are score tiles stored?
4. Is the score matrix materialized globally?
5. How is softmax computed incrementally?
6. Where is running output stored?
7. How much off-chip memory traffic is avoided?
8. Which operations are fused?
9. Which hardware memory level is being used?

---

## Core Mental Model

FlashAttention:

Attention
 → tile Q/K/V
 → load tiles close to compute
 → compute score tile
 → online softmax
 → multiply with V tile
 → accumulate output
 → discard temporary tile
 → repeat

Same attention mathematics.
Different and much more memory-efficient execution.
