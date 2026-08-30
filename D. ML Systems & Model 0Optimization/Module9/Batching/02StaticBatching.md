# Module 8 — Lesson 2: Static Batching

## Core Idea

Static batching:

Collect a fixed set of requests
→ form a batch
→ execute them together
→ batch remains fixed during execution.

Example:

R1 ─┐
R2 ─┤
R3 ─┤ → Model → NPU
R4 ─┘

---

## Why Batch?

Batching exposes parallelism and can improve:

- accelerator utilization
- throughput
- amortization of overhead

---

## Different Sequence Lengths

Example:

R1 = 5 tokens
R2 = 10 tokens
R3 = 20 tokens
R4 = 7 tokens

Dense batching may require padding to:

max_length = 20

Therefore:

actual tokens = 5 + 10 + 20 + 7 = 42

dense positions = 4 × 20 = 80

Some computation may therefore correspond to padding.

---

## Important

Masking padding does NOT necessarily mean that all
computation associated with padded positions disappears.

masking ≠ automatically eliminating computation

The actual effect depends on the implementation/kernel.

---

## Batching Window

A serving system may wait briefly to collect requests.

Example:

request arrives
→ wait up to N milliseconds
→ form batch
→ execute

Larger waiting windows can create larger batches,
but increase request latency.

---

## Main Static-Batching Limitation

LLM requests have different:

- arrival times
- prompt lengths
- generation lengths
- completion times

Therefore requests do not naturally finish together.

Example:

R1 ─────────────────→
R2 ───────────→
R3 ─────→
R4 ───────────────→

A fixed batch can contain idle slots when some requests finish.

---

## KV Cache Connection

Each request has its own KV cache.

When a request finishes:

its KV cache should ideally be released
and its memory reused.

Static batching does not naturally handle this
dynamic request lifecycle.

---

## Static Batching Is Useful For

- offline inference
- predictable workloads
- similar sequence lengths
- controlled batch sizes
- bulk processing

It is simple and can provide good accelerator utilization.

---

## Reasoning Checklist

When seeing static batching, ask:

1. How are requests grouped?
2. What happens when lengths differ?
3. How much padding is introduced?
4. What happens when one request finishes early?
5. Can its slot be reused?
6. What happens to its KV cache?
7. How long do new requests wait?
8. Is throughput improvement worth the latency/memory cost?

---

## Key Limitation

Static batching assumes:

same batch
→ execute together
→ finish together

Real LLM serving:

requests arrive and finish independently.

This motivates Dynamic Batching and eventually
Continuous Batching.
