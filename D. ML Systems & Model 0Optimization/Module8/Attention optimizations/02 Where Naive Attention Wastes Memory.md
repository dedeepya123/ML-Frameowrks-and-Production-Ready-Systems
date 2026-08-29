# Module 7 — Lesson 2: Where Naive Attention Wastes Memory

## Core Problem

Attention:

S = QKᵀ
P = softmax(S)
O = PV

Q/K/V:
[N, d]

S/P:
[N, N]

The N×N intermediate grows quadratically with sequence length.

---

## Why This Is Expensive

Naive attention may:

QKᵀ
 ↓
write S to memory
 ↓
read S
 ↓
softmax
 ↓
write P
 ↓
read P
 ↓
P × V

The same large intermediate is repeatedly moved through memory.

---

## Memory Hierarchy

Registers
 ↓
On-chip SRAM/cache
 ↓
Off-chip memory

Closer to compute:
- smaller
- faster
- useful for reuse

Off-chip:
- larger
- higher access cost
- limited bandwidth

Optimization goal:
keep useful data close to compute and minimize unnecessary
off-chip memory traffic.

---

## Why Sequence Length Hurts

Attention matrix:

N × N

For N = 4096:

N² = 16,777,216 elements

At FP16:
≈ 32 MB per attention matrix per head

Actual total memory depends on:
- batch
- number of heads
- layers
- dtype
- implementation

---

## Arithmetic Intensity Connection

Arithmetic intensity:

AI = FLOPs / bytes moved

Naive attention can suffer from expensive intermediate
memory movement.

An optimization can improve performance by reducing bytes
moved, even if the fundamental attention FLOPs remain similar.

---

## Important Distinction

FlashAttention does NOT eliminate the fundamental
O(N²) attention computation.

Its major advantage is reducing:
- intermediate memory usage
- off-chip memory traffic
- unnecessary materialization

---

## Core Insight

The mathematical computation:

QKᵀ → softmax → PV

does not require the entire N×N score/probability matrix
to be materialized in slower memory.

This observation leads to memory-efficient attention.

---

## Reasoning Pattern

Model:
Attention(Q,K,V)

Tensor:
Q/K/V → N×N score matrix

Compute:
QKᵀ → softmax → PV

Memory:
large intermediates repeatedly read/written

Runtime:
multiple stages/kernels

Hardware:
off-chip ↔ on-chip ↔ compute

Optimization target:
reduce unnecessary data movement.
