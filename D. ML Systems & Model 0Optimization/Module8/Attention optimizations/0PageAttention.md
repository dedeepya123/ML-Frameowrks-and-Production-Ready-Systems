# Module 7 — Lesson 6: Paged Attention

## Core Idea

Paged Attention addresses a SYSTEMS problem:

How should the growing KV cache be physically managed
when many variable-length requests are being served?

Instead of requiring each request's KV cache to occupy one
large contiguous memory region, KV cache is divided into
fixed-size blocks/pages.

---

## Contiguous KV Cache

Request A:

[ KV KV KV KV KV KV KV KV ]

Problems with large-scale dynamic serving:

- requires contiguous allocation
- dynamic growth can be difficult
- memory fragmentation
- inefficient reservation
- difficult memory reuse across requests

---

## Paged/Block-Based KV Cache

A sequence is divided into logical KV blocks.

Example:

Block size = 4 tokens

tokens 0–3  → logical block 0
tokens 4–7  → logical block 1
tokens 8–11 → logical block 2

Physical blocks do not have to be contiguous.

Example:

logical block 0 → physical block 7
logical block 1 → physical block 2
logical block 2 → physical block 9

A block table maintains this mapping.

---

## Why It Helps

Paged KV cache enables:

- dynamic allocation
- non-contiguous physical storage
- better memory utilization
- reduced fragmentation
- block reuse
- easier management of variable-length requests
- efficient memory pooling

When a request finishes:

its blocks can be returned to the pool
and reused by another request.

---

## Connection to Continuous Batching

Serving systems have requests that:

- arrive at different times
- have different context lengths
- generate different numbers of tokens
- finish at different times

Therefore KV memory requirements continuously change.

Paged allocation handles this dynamic environment better
than large contiguous allocations.

---

## Important Distinction

FlashAttention:

Focus = HOW to COMPUTE attention efficiently.

Uses:
- tiling
- online softmax
- avoiding large attention intermediates
- reducing memory traffic

Paged Attention:

Focus = HOW to MANAGE/ACCESS the KV CACHE efficiently.

Uses:
- KV blocks/pages
- block tables
- dynamic allocation
- memory pools
- non-contiguous physical storage

They can be used together.

---

## Framework vs Runtime

Framework/model level:

past_key_values
DynamicCache
cache_position
logical KV cache

Runtime/system level:

KV blocks
block allocation
block table
physical addresses
memory pool
memory reuse

The framework abstraction does not imply one universal
physical implementation.

The runtime/serving system may choose its own cache
management strategy.

---

## Mental Model

Attention compute:

Q + K + V
    ↓
efficient tiled computation
    ↓
FlashAttention-style optimization

KV storage:

requests
    ↓
KV blocks
    ↓
memory pool
    ↓
block table
    ↓
physical memory

---

## Best Explanation

"Paged Attention is a systems-oriented approach where the KV
cache is divided into fixed-size blocks that can be stored
non-contiguously in a memory pool. A mapping table tells the
attention kernel where each logical block resides physically.
This makes dynamic KV-cache allocation and reuse much more
efficient for serving many variable-length requests."

---

## Reasoning Questions

When reading a runtime implementation, ask:

1. How is KV cache physically allocated?
2. Does each request require contiguous memory?
3. How are KV blocks represented?
4. How is logical → physical mapping maintained?
5. When does a new block get allocated?
6. When are blocks released?
7. Can released blocks be reused?
8. How does the attention kernel locate the blocks?
9. Where are the blocks stored physically?
10. How are they moved to the compute units?

These questions connect model-level KV cache knowledge
to runtime-level memory management.
