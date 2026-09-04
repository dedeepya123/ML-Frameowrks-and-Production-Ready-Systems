# Module 9 — Lesson 6: PCIe

## Core idea

PCIe is a high-speed communication/interconnect mechanism used to
connect the CPU/system with devices such as GPUs, NPUs, NICs and storage.

For ML Systems, the important concept is not the PCIe protocol itself,
but the cost of moving data across the interconnect.

## Host vs Device Memory

Host memory:
    System/CPU RAM

Device memory:
    Memory associated with or accessible by an accelerator.

Conceptually:

CPU/System RAM
      ↓
   Interconnect
      ↓
Accelerator
      ↓
Device Memory

## Communication cost

Approximate:

T_comm ≈ latency + data_size / bandwidth

Large transfers:
    bandwidth becomes important.

Small/frequent transfers:
    latency can become important.

## Why PCIe matters for ML

An accelerator may be extremely fast, but data still has to move.

If:

    Compute = 10 ms
    Transfer = 100 ms

then improving accelerator compute won't solve the main bottleneck.

## Relation to parallelism

Tensor Parallelism:
    devices exchange tensors/results.

Pipeline Parallelism:
    devices exchange activations between stages.

The communication path can involve PCIe or another
higher-performance accelerator interconnect depending on the system.

## Important distinction

PCIe:
    communication path

Device memory:
    where data resides

Do not confuse the two.

Analogy:

    Memory = warehouse
    PCIe = road

A large warehouse with a narrow road can still create traffic.

## Batching

Many small transfers may incur more overhead.

Fewer larger transfers can use bandwidth more efficiently,
but batching also introduces latency/memory/scheduling tradeoffs.

## NPU connection

Conceptually:

CPU
 ↓
Runtime
 ↓
Interconnect
 ↓
NPU
 ↓
NPU-accessible memory

Important optimization questions:

- Where is the data?
- How much data crosses CPU ↔ accelerator?
- How often?
- Can data remain resident?
- Can communication overlap with computation?
- Is the interconnect the bottleneck?

## What NOT to study deeply right now

Not required for current ML Systems goal:

- PCIe electrical details
- packet/protocol internals
- encoding mechanisms
- transaction-layer details
- exhaustive generation specifications

## Mental model

When you see PCIe, ask:

"What data is crossing the link,
how much, how often,
and is the link fast enough?"
