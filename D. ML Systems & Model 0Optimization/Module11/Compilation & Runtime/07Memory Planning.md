# Module 10 — Lesson 7: Memory Planning

## Core idea

**Memory planning = deciding how much memory tensors need, where their storage lives, how long it is needed, and when storage can be reused.**

## Key concept: tensor lifetime

For:

```text
A → B → C
```

if B is the last operation that uses A:

```text
A created
   ↓
used by B
   ↓
A dies
   ↓
A's buffer can potentially be reused
```

## Why memory planning?

* reduce peak memory usage
* reuse buffers
* avoid unnecessary allocations
* make execution predictable
* plan memory placement and transfers

## Shape → memory connection

```text
Tensor shape
     ↓
Tensor size
     ↓
Memory requirement
     ↓
Memory plan
```

Static/predictable shapes make this easier.

## Fusion → memory connection

Without fusion:

```text
A → write intermediate → B → output
```

With fusion:

```text
A → B → output
```

The intermediate may never require a global-memory buffer.

Therefore:

> Fusion can reduce memory traffic and memory requirements.

## KV cache distinction

Temporary activations have short lifetimes:

```text
create → use → release/reuse
```

KV cache has a long lifetime across inference steps:

```text
token 1 ──────────────┐
token 2 ──────────────┤
token 3 ──────────────┤
...                    │
token N ──────────────┘
```

Therefore KV cache requires different runtime memory-management strategies.

## Dynamic shapes

Dynamic shapes make exact memory planning harder because tensor sizes are not always known beforehand.

Possible strategies include:

* dynamic allocation
* memory pools
* bounded shapes
* shape specialization
* conservative reservations

## QAIRT / Gemma4 connection

Predictable dimensions such as:

```text
ARN = 521
GCL = 15527
```

and explicit mask/cache-related shapes give the compilation/runtime system stronger information for memory planning.

The adaptation makes execution structure more explicit; the compiler/runtime performs the actual memory planning.

## Important distinction

```text
Memory Planning
    ↓
Where/how long tensors use memory

FlashAttention
    ↓
Algorithmic reduction of attention memory traffic

KV Cache Management
    ↓
Managing long-lived inference state
```

These are related but different concepts.

## Mental model

```text
Graph
  ↓
Shapes + dependencies
  ↓
Tensor lifetimes
  ↓
Memory planning
  ↓
Buffer allocation/reuse
  ↓
Kernel execution
  ↓
Runtime
  ↓
Hardware
```

### One-line takeaway

> **Memory planning uses graph structure, shapes, and tensor lifetimes to minimize and organize the memory required during execution.**
