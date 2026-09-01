# Module 8 — Lesson 6: KV Cache Management

## Core Idea

At model level:

KV cache = model state.

At serving level:

KV cache = a large, dynamic, shared memory resource
used by many concurrent requests.

---

## Why KV Cache Becomes a Systems Problem

Many requests may be active simultaneously:

R1 → KV1
R2 → KV2
R3 → KV3
...
R100 → KV100

Each request:

- has different KV requirements
- grows during decode
- finishes at a different time
- releases memory when finished

Therefore KV cache requires memory management.

---

## Naive Contiguous Allocation

Each request receives one contiguous region:

R1 → [10 MB]
R2 → [20 MB]
R3 → [50 MB]

Problem:

KV cache grows dynamically.

If the adjacent memory is occupied, the cache may need
to be relocated/copied.

---

## Fragmentation

Example:

[R1][FREE][R3][R4][FREE][R6]

There may be enough total free memory but not enough
contiguous memory for a large request.

Therefore:

total free memory != largest available contiguous block

---

## Block/Page-Based Allocation

Instead of allocating one large contiguous region,
divide KV memory into fixed-size blocks/pages.

Example:

R1 → [B3, B8, B11, B17]

Blocks do not need to be physically contiguous.

This makes allocation and reuse much easier.

---

## Block Table

The system maintains a logical-to-physical mapping.

Example:

Logical block → Physical block

0 → 12
1 → 4
2 → 31
3 → 8

The request sees a logical sequence of KV blocks while
the physical blocks may exist anywhere in the KV memory pool.

---

## Request Lifecycle

Request arrives
→ allocate KV blocks
→ prefill
→ KV grows during decode
→ allocate additional blocks
→ request finishes
→ release blocks
→ blocks return to free pool
→ new requests can reuse them

---

## PagedAttention Connection

PagedAttention from Module 7 is related to how attention
accesses KV stored in blocks.

KV Cache Management asks:

- How are blocks allocated?
- How are they tracked?
- How are they freed?
- How are they reused?

Paged Attention asks:

- How does attention access the KV blocks needed for
  the sequence?

---

## Logical vs Physical KV Cache

Model/framework level:

request
→ cache object
→ K/V tensors

Serving level:

request
→ block table
→ physical KV blocks

Important distinction:

logical model state != physical memory representation

---

## KV Cache Sharing

Requests with shared prefixes may potentially share
KV blocks.

Conceptually:

       shared prefix
          ↓
       [B1][B2][B3]
         ↙       ↘
       R1         R2

This can reduce duplicated KV memory.

This is related to prefix caching/KV reuse.

---

## When KV Memory Is Full

If no KV blocks are available, the system may need to:

- delay admission
- queue the request
- reject the request
- reclaim memory
- use other system-specific strategies

Therefore KV-cache capacity can become an admission-control
constraint.

---

## Connection to Batching

More concurrent requests
→ more KV cache
→ more memory consumption
→ KV capacity affects scheduling

Therefore:

batch size is not only a compute decision;
it is also a memory decision.

---

## Connection to Hardware

KV cache can create significant memory traffic.

KV cache
→ memory accesses
→ bandwidth pressure
→ decode performance

This connects:

Module 2:
memory hierarchy / bandwidth / compute-vs-memory

Module 4:
KV cache

Module 8:
KV-cache management

---

## Serving Architecture

User Requests
      ↓
Request Queue
      ↓
Scheduler
      ↓
Prefill / Decode
      ↓
Model
      ↓
Runtime
      ↓
NPU
      ↓
KV Cache

                 ↑
                 │
          KV Cache Manager
          allocate/free/reuse

---

## Reasoning Framework

When reading a production KV-cache implementation,
ask:

1. What is the logical unit?
   → request/sequence

2. What is the physical allocation unit?
   → block/page

3. Who maintains request → block mapping?

4. How does KV cache grow?
   → allocate additional blocks

5. How is memory reclaimed?
   → free blocks when request finishes

6. Can blocks be shared?
   → prefix/KV sharing

7. What happens when blocks run out?
   → queue/reject/evict/offload/etc.

---

## Important Mental Model

Do NOT think:

continuous batching = continuously increasing batch size.

Think:

continuous batching =
managing a changing set of active sequences.

Each active sequence has:

logical state
+
KV-cache memory state.

---

## Key Statement

At model level:

"KV cache is model state."

At serving level:

"KV cache is a scarce memory resource shared by many
concurrent sequences."

---

## Next

Lesson 7 — Scheduler → Runtime → Model Flow

Trace one request end-to-end and clearly separate:

Serving System
Framework
Runtime
Model
Kernel
Hardware
