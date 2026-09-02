# Module 8 — Lesson 7: Scheduler → Runtime → Model Flow

## Core Idea

Production inference has multiple layers:

Serving
→ Scheduler
→ KV/Batch Management
→ Runtime
→ Model
→ Kernels
→ Hardware

Each layer answers a different question.

---

## Layer Responsibilities

Serving:
Which request/server should handle the request?

Scheduler:
What work should execute now?

Batching:
Which requests should execute together?

KV Cache Manager:
Where is request state stored?

Runtime:
How should the computation execute?

Model:
What computation should be performed?

Kernel:
How is an operation implemented efficiently?

Hardware:
Where/how is computation physically executed?

---

## One Request

User
→ request queue
→ scheduler
→ runtime
→ model
→ kernels
→ NPU/GPU
→ output
→ scheduler
→ next iteration

---

## Decode Loop

Request needs next token
→ scheduler selects it
→ KV state located
→ runtime executes model
→ NPU executes kernels
→ logits produced
→ next token generated
→ KV updated
→ scheduler continues request

---

## Prefill/Decode

Scheduler may choose between:

Prefill:
new requests processing prompts

Decode:
existing requests generating tokens

The choice affects:

- TTFT
- inter-token latency
- throughput
- accelerator utilization
- KV memory

---

## Millions of Users

One inference engine does NOT serve millions
of users directly.

Instead:

Millions of users
→ load balancing
→ many inference workers
→ many runtimes
→ many accelerators

This is horizontal scaling.

Within each worker:

many concurrent requests
→ batching
→ scheduling
→ shared accelerator utilization

---

## Two Scaling Dimensions

Across machines:

more users
→ more workers/accelerators

Within a worker:

more concurrent requests
→ batching/scheduling
→ better accelerator utilization

---

## Model Replication

A model may be replicated:

Model
→ Worker 1
→ Worker 2
→ Worker 3
→ ...

For very large models, one model copy may instead
span multiple accelerators.

This leads to:

Module 9 — Parallelism & Communication

---

## Four-Level Mental Model

LEVEL 1 — SERVING

Users
→ requests
→ scheduler
→ batching
→ KV management

LEVEL 2 — MODEL

Input
→ embedding
→ attention
→ MLP
→ LM head

LEVEL 3 — RUNTIME

Graph
→ operators
→ fusion
→ memory planning
→ kernels

LEVEL 4 — HARDWARE

Kernel
→ NPU/GPU
→ compute units
→ on-chip memory
→ DRAM

---

## Important Systems Thinking

When seeing a model/runtime operation, ask:

Model:
What mathematical operation is this?

Framework:
How is it represented?

Runtime:
How is it represented/executed?

Kernel:
What implementation performs it?

Hardware:
Where does data live and where does computation execute?

Serving:
How does this behave with many concurrent requests?

---

## Key Statement

Large-scale LLM serving is not a completely different
inference concept.

It is the same core model/runtime/batching ideas
embedded inside a distributed system with:

- many workers
- many accelerators
- routing
- scheduling
- batching
- KV management
- distributed communication

---

## Module 8 Summary

We learned:

1. Why batching exists
2. Static batching
3. Dynamic batching
4. Continuous batching
5. Prefill/decode scheduling
6. KV cache management
7. Scheduler → Runtime → Model flow

Core mental model:

Users
→ Queue
→ Scheduler
→ Batch
→ KV Management
→ Runtime
→ Model
→ Kernels
→ Accelerator
→ KV state
→ Next iteration
