# Module 7 — Lesson 3: Memory-Efficient Attention

## Goal

Compute the same attention result without materializing
the full N×N attention matrix.

---

## Naive Attention

QKᵀ
 ↓
materialize N×N scores
 ↓
softmax
 ↓
materialize probabilities
 ↓
× V

Large intermediate memory and memory traffic.

---

## Memory-Efficient Idea

Process attention in blocks/tiles.

Q block
K block
V block
   ↓
small score block
   ↓
softmax incrementally
   ↓
update output
   ↓
discard temporary block
   ↓
next block

The full N×N matrix never needs to be materialized.

---

## Online Softmax

Softmax normally requires information across
the complete row.

Instead of materializing the entire row, maintain
running information such as:

- running maximum
- normalization information
- running output

Blocks can therefore be processed incrementally.

---

## Complexity

Attention computation remains fundamentally:

O(N²d)

The optimization primarily reduces:

- intermediate memory
- memory traffic
- materialization

Memory can move from an O(N²) intermediate toward
O(Nd) plus working tiles, depending on implementation.

---

## Memory Hierarchy

Load small Q/K/V tiles:

Off-chip memory
      ↓
On-chip memory
      ↓
compute
      ↓
update output
      ↓
discard temporary data

Then process the next tile.

---

## Arithmetic Intensity

AI = FLOPs / bytes moved

Memory-efficient attention reduces expensive memory
movement and increases data reuse.

The goal is not necessarily fewer FLOPs.

The goal is:
same useful computation + less data movement.

---

## Fusion

Instead of:

QKᵀ
 ↓
memory
 ↓
softmax
 ↓
memory
 ↓
PV

An optimized attention kernel can coordinate:

load tiles
→ QKᵀ
→ online softmax
→ PV accumulation
→ final output

This avoids unnecessary intermediate writes/reads.

---

## Tradeoff

Reducing memory traffic can require:

- more sophisticated kernels
- online statistics
- possible recomputation
- careful numerical handling

We trade some computation/complexity for reduced memory traffic.

---

## FlashAttention Connection

Memory-efficient attention is the general idea.

FlashAttention is a highly optimized implementation of
these principles.

Progression:

Naive Attention
→ identify memory problem
→ block/tile computation
→ online softmax
→ avoid N×N materialization
→ FlashAttention

---

## Core Principle

Do not ask only:

"How many FLOPs?"

Also ask:

"How much data moves between memory levels?"

And:

"Can I keep/reuse data close to the compute units?"
