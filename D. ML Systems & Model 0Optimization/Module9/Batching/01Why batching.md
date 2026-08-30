# Module 8 — Lesson 1: Why Batching?

## Core Question

Why process multiple inference requests together?

Because batching can:

- expose more parallelism
- improve accelerator utilization
- amortize overhead
- improve throughput

---

## Basic Idea

Without batching:

R1 → Model → Accelerator
R2 → Model → Accelerator
R3 → Model → Accelerator

With batching:

R1 ─┐
R2 ─┤
R3 ─┤ → Model → Accelerator
R4 ─┘

Multiple requests execute together.

---

## Why It Helps

Accelerators contain many parallel compute resources.

Small workloads may not use all of them.

Batching provides more independent work and can therefore
increase utilization and throughput.

---

## Batching Is Not Automatically Better

Increasing batch size can also cause:

- higher memory usage
- larger KV-cache requirement
- more memory traffic
- increased latency
- more scheduling complexity

Therefore:

Batch size ↑ does NOT automatically mean performance ↑.

---

## Throughput vs Latency

Latency:
Time required for an individual request.

Throughput:
Amount of work completed per unit time.

Batching generally targets higher throughput but may affect
individual-request latency.

---

## Static Batching

Wait for requests:

R1 R2 R3 R4
     ↓
Form fixed batch
     ↓
Run model

Simple, but less flexible for variable-length LLM requests.

---

## Why LLM Batching Is Harder

Requests have:

- different prompt lengths
- different generation lengths
- different arrival times
- different KV-cache sizes
- different completion times

During serving, some requests may be prefilling while
others are decoding.

Prefill and decode also have different performance
characteristics.

---

## Batching vs Scheduling

Batching:
Which requests execute together?

Scheduling:
When and how should work execute?

They work together in a serving system.

---

## Systems-Level Flow

Requests
   ↓
Scheduler
   ↓
Batch formation
   ↓
Runtime
   ↓
Model
   ↓
NPU

The system tries to optimize:

- throughput
- latency
- hardware utilization
- memory utilization
- KV-cache capacity
- cost

---

## Reasoning Checklist

When someone says "we increased batch size", ask:

1. Did parallelism increase?
2. Did hardware utilization increase?
3. Did throughput improve?
4. How much additional KV memory is required?
5. Did memory bandwidth become a bottleneck?
6. Did latency increase?
7. Did useful performance actually improve?

---

## Central Mental Model

Batching is a tradeoff:

Batch size ↑
→ parallelism ↑
→ potential throughput ↑
→ potential accelerator utilization ↑
→ KV/memory requirements ↑
→ latency may ↑

The correct batch size depends on the workload,
hardware, memory capacity, and latency requirements.

---

## Connection to Previous Modules

Module 2:
Compute vs memory and bottleneck reasoning.

Module 4:
KV-cache memory grows with active requests.

Module 7:
Efficient attention and KV-cache execution.

Module 8:
How to efficiently execute MANY requests together.

---

## Target Skill

Do not think:

"Large batch = faster."

Think:

"Does increasing the batch expose useful parallelism,
and is the resulting gain in throughput worth the additional
memory, bandwidth, and latency cost?"
