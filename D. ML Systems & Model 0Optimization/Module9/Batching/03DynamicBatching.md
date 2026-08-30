# Module 8 — Lesson 3: Dynamic Batching

## Core Idea

Dynamic batching forms batches based on requests that are
available at runtime rather than relying on a permanently
fixed batch.

Requests:
R1 R2 R3
   ↓
scheduler
   ↓
batch [R1,R2,R3]

A later batch may contain a completely different set of
requests.

---

## Why Dynamic Batching?

Real requests arrive at different times.

Dynamic batching allows the system to adapt to the current
workload instead of requiring predetermined batches.

---

## Typical Mechanism

Request arrives
→ enters queue
→ scheduler observes available requests
→ waits briefly OR executes
→ forms batch
→ runs model

Example policy:

max_batch_size = 4
max_wait_time = 5 ms

Execute when:
- batch reaches 4 requests, OR
- waiting time reaches 5 ms

---

## Core Tradeoff

Wait for more requests:

+ larger batch
+ potentially better throughput
- higher waiting latency

Execute immediately:

+ lower latency
- smaller batch
- potentially lower accelerator utilization

Therefore the scheduler balances:

Latency ↔ Throughput

---

## Different Sequence Lengths

Dynamic batching does NOT automatically solve variable
sequence lengths.

Example:

R1 = 10 tokens
R2 = 100 tokens
R3 = 500 tokens

A dense implementation may still require padding.

Dynamic batching answers:

"Which requests should execute together?"

It does not automatically answer:

"How should differently shaped requests be executed efficiently?"

---

## Why LLM Serving Is Harder

Traditional inference:

request
→ one model execution
→ response

Autoregressive LLM:

request
→ model execution
→ token
→ model execution
→ token
→ model execution
→ ...

A request remains active across many model iterations.

Therefore normal batch formation is not enough.

---

## KV Cache Connection

Each active request owns/uses KV-cache state.

Example:

R1 → KV A
R2 → KV B
R3 → KV C

If R2 finishes:

R2 finishes
→ reclaim KV memory
→ new request R4 can potentially enter

Active set changes:

Before:
[R1,R2,R3]

After:
[R1,R3,R4]

This leads naturally toward continuous batching.

---

## Dynamic vs Continuous Batching

Dynamic:

requests arrive
→ form batch
→ execute
→ form next batch

Continuous:

active requests
→ execute iteration
→ finished requests leave
→ new requests enter
→ execute next iteration
→ repeat

The key distinction is whether the active batch can change
while ongoing autoregressive generation continues.

---

## Systems View

Users
 ↓
Request Queue
 ↓
Scheduler
 ↓
Dynamic Batcher
 ↓
Runtime
 ↓
Model
 ↓
NPU

The scheduler may consider:

- request waiting time
- batch size
- sequence length
- KV-cache capacity
- latency targets
- accelerator capacity

---

## Important Reasoning Questions

When looking at a batching system, ask:

1. When are batches formed?
2. How long can requests wait?
3. What determines batch size?
4. Can requests enter after execution starts?
5. Can requests leave before the batch finishes?
6. How are different sequence lengths handled?
7. How is KV cache allocated/reclaimed?
8. What happens when memory is full?
9. What is being optimized: latency, throughput, or both?
10. Does the accelerator actually benefit from the larger batch?

---

## Key Mental Model

Static:
fixed group → execute

Dynamic:
runtime request queue → scheduler chooses group → execute

Continuous:
active set changes every generation iteration

---




## Central Question

Dynamic batching is fundamentally about:

"Given the requests currently waiting and running, should
the system execute now or wait for more work?"

The answer is a tradeoff between:

- latency
- throughput
- memory
- accelerator utilization

---
``` text
Static Batching
│
├── fixed group
├── simple
└── inefficient when request lifetimes differ
          ↓
Dynamic Batching
│
├── batches formed at runtime
├── adapts to arrival rate
├── improves batching opportunity
└── still treats execution in batch-oriented chunks
          ↓
Continuous Batching
│
├── active requests can change every iteration
├── finished requests leave
├── new requests enter
└── much better suited to autoregressive LLM serving
```

## Next

Lesson 4 — Continuous Batching

Focus:
How can new requests enter and completed requests leave
while other requests are still generating tokens?
