# Module 9 — Lesson 8: Compute vs Communication Tradeoffs

## Core idea

Parallelism does not automatically produce linear speedup.

A distributed workload has:

    Compute
    + Communication
    + Synchronization

Therefore:

    T_total ≈ T_compute + T_communication + T_synchronization

## Ideal vs real scaling

Ideal:

    N devices → N× speedup

Real:

    N devices → less than N× speedup

because communication and synchronization introduce overhead.

## Parallel efficiency

    Efficiency = Speedup / Number of devices

Example:

    2 GPUs
    Speedup = 1.67×

    Efficiency = 1.67 / 2
               ≈ 83.5%

## Communication can dominate

As the number of devices increases:

    Compute per device ↓
    Communication overhead may ↑

Eventually:

    additional devices → diminishing returns

A workload can become communication-bound.

## Parallelism-specific costs

Data Parallelism:
    primarily distributes independent requests for inference.
    Good for throughput when model fits on each device.

Tensor Parallelism:
    distributes computation within layers.
    Communication can happen frequently between devices.

Pipeline Parallelism:
    distributes layers into stages.
    Activations must move between stages.

## Communication overlap

Instead of:

    Compute → Communication → Compute

we can sometimes achieve:

    Compute ─────────────
        Communication ───

Communication happens concurrently with useful computation.

This can hide part of the communication cost.

## Load balancing

Poorly balanced work causes devices to wait.

Example:

    10 ms / 10 ms / 40 ms / 10 ms

The 40 ms stage becomes the bottleneck.

Good partitioning considers:

    compute
    memory
    communication
    synchronization

## Amdahl's Law intuition

If a significant fraction of execution cannot be accelerated,
optimizing the remaining portion has limited end-to-end benefit.

In distributed ML:

    compute
    communication
    synchronization

all contribute to total time.

## Three reasons for parallelism

1. Capacity
   Model does not fit on one device.

2. Throughput
   Need to process many requests.

3. Compute / latency
   One workload takes too long.

The appropriate parallelism depends on the bottleneck.

## ML Systems debugging questions

If scaling is worse than expected, ask:

1. What parallelism is being used?
2. What is being partitioned?
3. What data is communicated?
4. How many bytes are transferred?
5. How frequently?
6. Which communication primitive is used?
7. What interconnect is used?
8. What is the hardware topology?
9. Are devices synchronized?
10. Are stages balanced?
11. Can communication overlap with computation?
12. Is the workload compute-, memory-, or communication-bound?

## Key principle

Parallelism is not free.

The goal is:

    Benefit from parallelism
        >
    Cost of communication + synchronization
