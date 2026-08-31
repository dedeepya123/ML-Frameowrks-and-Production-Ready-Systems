# Module 8 — Lesson 4: Continuous Batching

## Core Idea

Continuous batching allows the set of active requests to
change while autoregressive generation is still running.

Completed requests leave.
New requests can enter.
Remaining requests continue generating.

---

## Example

Iteration 1:
[R1 R2 R3]

Iteration 2:
[R1 R2 R3]

R2 finishes.

Iteration 3:
[R1 R3 R4]

R4 has joined while R1 and R3 are still generating.

---

## Why It Helps

Static batching may produce:

[R1 R2 R3]

R2 finishes:

[R1 EMPTY R3]

Potentially wasted capacity.

Continuous batching can replace completed work:

[R1 R3 R4]

This can keep the accelerator better utilized and improve
throughput.

---

## Why LLMs Need It

Autoregressive generation repeatedly executes the model:

request
→ token
→ model
→ token
→ model
→ token
→ ...

A request remains active for many iterations.

Different requests have different generation lengths.

Therefore a fixed batch is inefficient.

---

## KV Cache Connection

Each active request has KV-cache state.

Example:

R1 → KV1
R2 → KV2
R3 → KV3

If R2 finishes:

R2 → finished
KV2 → reclaim

New request R4 can potentially use the reclaimed/available
memory.

Therefore:

request lifecycle
↔
KV-cache lifecycle

---

## Important Mental Model

Do not think:

"I have one fixed batch."

Think:

"I have a set of currently active sequences."

Every iteration:

active requests
→ scheduler
→ choose next execution set
→ model
→ generate tokens
→ remove finished requests
→ admit new requests
→ repeat

---

## Continuous Batching vs Dynamic Batching

Dynamic batching:

requests arrive
→ form batch
→ execute
→ next batch

Continuous batching:

active requests
→ execute generation iteration
→ completed requests leave
→ new requests enter
→ execute next iteration

The key difference is that the active set can change while
ongoing autoregressive generation continues.

---

## Prefill vs Decode

A newly arriving request usually needs:

prompt
→ prefill
→ KV cache creation
→ decode

Existing requests may already be decoding.

Therefore a serving system may simultaneously have:

Prefill work:
R4

Decode work:
R1, R2, R3

Prefill and decode have different performance
characteristics.

This motivates Prefill/Decode Scheduling.

---

## Systems View

Request Queue
     ↓
Scheduler
     ↓
Active Requests
     ↓
Runtime
     ↓
Model
     ↓
NPU

After each generation iteration:

- finished requests are removed
- new requests may be admitted
- KV memory is reclaimed/allocated
- next active set is formed

---

## Connection to HF generate()

Single-request HF generation conceptually:

while not finished:
    model(...)
    choose token
    update cache
    check stopping condition

Continuous batching generalizes this:

while requests exist:
    scheduler chooses active requests
    model executes next iteration
    update KV state
    remove finished requests
    admit new requests

---

## Important Distinction

Continuous batching is a logical serving strategy.

Its physical implementation can use:

- slots
- fixed buffers
- padding
- dynamic shapes
- block tables
- specialized kernels

depending on the runtime.

Logical behavior != physical implementation.

---

## Key Tradeoffs

Continuous batching can improve:

- throughput
- accelerator utilization

But requires careful management of:

- KV-cache memory
- scheduling
- prefill/decode interaction
- latency
- batch composition
- runtime/kernel constraints

---

## Central Mental Model

Static:
fixed group → execute

Dynamic:
runtime forms batches based on arrivals

Continuous:
active request set changes every generation iteration

---

## One-Sentence Definition

Continuous batching is a serving strategy where completed
requests can leave and new requests can enter between
generation iterations, allowing the accelerator to remain
productively occupied while different requests generate
tokens at different rates.

---

text

Why batching?
      ↓
More parallelism / throughput

Static batching
      ↓
Fixed group
      ↓
Problem: requests have different lifetimes

Dynamic batching
      ↓
Form batches based on current arrivals
      ↓
Problem: LLM requests remain active for many iterations

Continuous batching
      ↓
Active set changes every generation iteration
      ↓
Problem: now we must manage prefill/decode + KV memory

Prefill/Decode Scheduling
      ↓
Decide how different workload types coexist

KV Cache Management
      ↓
Manage memory for all active requests

Scheduler → Runtime → Model
      ↓
Full serving-system picture
## Next

Lesson 5 — Prefill/Decode Scheduling

Question:

How should a serving system schedule new prompt-prefill
work together with existing decode work?
