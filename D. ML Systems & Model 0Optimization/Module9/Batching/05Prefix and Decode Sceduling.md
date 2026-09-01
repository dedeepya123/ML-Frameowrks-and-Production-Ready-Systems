# Module 8 — Lesson 5: Prefill/Decode Scheduling

## Core Idea

A serving system may have two types of work:

Prefill:
new requests processing their prompts

Decode:
existing requests generating the next token

The scheduler decides how to balance these workloads.

---

## Example

Active:

R1 R2 R3 → decode

Waiting:

R4 R5 → prefill

Scheduler must decide what should execute next.

---

## Prefill

Prompt tokens are processed to build the KV cache.

Example:

R4 = 1000 prompt tokens
→ Transformer
→ KV cache for R4

Prefill exposes substantial parallel computation and can
often utilize accelerator compute efficiently.

---

## Decode

Existing requests generate the next token.

Example:

R1 → token
R2 → token
R3 → token

Decode repeatedly accesses the growing KV cache.

It can therefore become strongly influenced by:

- memory bandwidth
- KV-cache reads
- memory capacity

---

## Main Scheduling Tradeoff

Prioritize prefill:

+ new requests progress
+ lower TTFT

- existing decode requests may wait
- inter-token latency can increase

Prioritize decode:

+ existing users continue receiving tokens
+ better token responsiveness

- new requests wait longer
- TTFT increases

Therefore the scheduler balances:

TTFT
↔
inter-token latency
↔
throughput
↔
accelerator utilization

---

## TTFT

Time To First Token:

request arrives
→ waiting
→ prefill
→ first generated token

TTFT measures the initial responsiveness of a new request.

---

## Inter-token Latency

For an active request:

token N
→ token N+1

The time between generated tokens affects the perceived
generation responsiveness.

---

## Why Request Size Matters

A small prompt and a huge prompt are not equivalent.

Example:

R4 = 5 tokens
R5 = 50,000 tokens

Blindly prioritizing large prefills can delay many existing
decode requests.

Scheduler should reason about amount of work, not merely
request count.

---

## KV Cache Constraint

Prefill creates KV-cache state.

Therefore admitting a new request increases memory usage.

Scheduler must consider:

available KV memory
+
expected KV growth
+
currently active requests

Scheduling is therefore also a memory-capacity problem.

---

## Chunked Prefill

Large prefills can conceptually be divided into smaller chunks.

Example:

8000-token prompt:

2000
→ decode existing requests
→ 2000
→ decode existing requests
→ 2000
→ ...

This allows prefill work and decode work to be interleaved.

Goal:

balance new-request TTFT with existing-request
inter-token latency.

---

## Connection to Continuous Batching

Continuous batching:

active request set can change between iterations.

Prefill/decode scheduling:

decides what type and amount of work should participate
in the next execution.

Together they create a more intelligent LLM serving loop.

---

## Serving Loop

while server is running:

    remove finished requests

    admit waiting requests if possible

    determine prefill work

    determine decode work

    execute next workload

    update KV/cache state

    repeat

---

## Architectural Boundary

Model level:

Q/K/V
Attention
MLP
KV cache
LM head

Runtime level:

operators
graphs
kernels
memory
NPU execution

Serving level:

requests
queue
scheduler
batching
KV allocation

Hardware level:

NPU compute
memory bandwidth
on-chip memory
DRAM

Flow:

Serving
 ↓
Runtime
 ↓
Model
 ↓
Kernels
 ↓
NPU

---

## Reasoning Questions

When analyzing a serving scheduler, ask:

1. Which requests are waiting?
2. Which requests are decoding?
3. How large are their prompts?
4. How much KV memory is available?
5. What is the TTFT requirement?
6. What is the inter-token latency requirement?
7. Is the accelerator compute-bound or memory-bound?
8. How much work should be admitted?
9. Should prefill be chunked?
10. What should execute next?

---

## One-Sentence Definition

Prefill/decode scheduling is the process of balancing
prompt-prefill work for new requests with token-decode work
for existing requests while considering latency,
throughput, accelerator utilization, and KV-cache capacity.

---

## Next

Lesson 6 — KV Cache Management

Question:

Once many requests are continuously entering and leaving
the system, how do we actually allocate, track, reclaim,
and reuse their KV-cache memory?
