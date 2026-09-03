# Module 9 — Lesson 5: Communication Cost

## Core idea

Distributed execution introduces communication because computation/data
is spread across multiple devices.

Parallelism does not automatically give linear speedup.

## Communication cost

Approximate model:

T_comm ≈ α + S/B

where:

α = communication latency/startup cost
S = amount of data transferred
B = effective bandwidth

Therefore:
- Large transfers → bandwidth matters.
- Small/frequent transfers → latency matters.
- Many communications → overhead can accumulate.

## Communication vs computation

Overall distributed execution can be thought of as:

T_total ≈ T_compute + T_communication + T_synchronization

If communication becomes large relative to computation,
the workload becomes communication-bound.

## Common communication patterns

Point-to-point:
    GPU0 → GPU1

All-Gather:
    each device has a shard
    everyone receives the complete data

All-Reduce:
    devices contribute values
    everyone receives the reduced result

Reduce-Scatter:
    reduce values and distribute portions of the result

## Relation to parallelism

Data Parallelism:
    replicas process different data.
    Training often requires gradient synchronization.

Tensor Parallelism:
    communication can include all-reduce, all-gather,
    reduce-scatter, depending on partitioning.

Pipeline Parallelism:
    primarily transfers activations between stages.

## Synchronization

Devices may need to wait for one another.

Example:

GPU0 = 5 ms
GPU1 = 8 ms
GPU2 = 20 ms
GPU3 = 7 ms

If synchronization is required, faster devices may wait for GPU2.

## Load balancing

Pipeline stages should be reasonably balanced.

Bad:
    10 ms / 10 ms / 40 ms / 10 ms

Better:
    20 ms / 20 ms / 20 ms / 20 ms

Partitioning should consider:
- memory
- compute
- communication
- synchronization

## Interconnect

Communication happens through links such as:
- PCIe
- NVLink
- accelerator-specific interconnects

Topology matters because not all device-to-device paths
necessarily have the same bandwidth/latency.

## Key mental model

Do not think:

    4 GPUs = 4x performance

Think:

    4 compute resources
    + finite-bandwidth communication
    + synchronization

## Key questions

When analyzing distributed inference/training, ask:

1. What is being partitioned?
2. What data must move?
3. How much data moves?
4. How frequently does it move?
5. What communication primitive is used?
6. What is the interconnect bandwidth/latency?
7. Do devices need synchronization?
8. Can communication overlap with computation?
9. Are workloads/stages balanced?
10. Is the system compute-bound or communication-bound?
