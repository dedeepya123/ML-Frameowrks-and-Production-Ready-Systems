# Module 9 — Lesson 7: NVLink / High-Speed Interconnects

## Core idea

Accelerator parallelism requires communication between devices.

A high-performance interconnect can reduce the cost of that communication.

NVLink is a high-bandwidth, low-latency interconnect designed for
communication between supported NVIDIA GPUs and related GPU systems.

## PCIe vs NVLink

PCIe:
    General-purpose system/device interconnect.

NVLink:
    High-performance accelerator interconnect.

Do not think of NVLink as memory.
It is a communication path.

## Why it matters for ML

Tensor Parallelism:
    Devices frequently exchange tensors/results.

Pipeline Parallelism:
    Devices exchange activations between stages.

Therefore:

Parallelism
    ↓
Communication
    ↓
Interconnect performance
    ↓
Distributed inference performance

## Communication cost

T_comm ≈ latency + data_size / bandwidth

For large transfers:
    bandwidth is important.

For small frequent transfers:
    latency can become important.

## Topology

Having multiple GPUs is not enough.

We also care about:

    Which devices are connected?
    How fast are the connections?
    What path does communication take?

Parallelization strategy should ideally match hardware topology.

## Important distinction

NVLink does not make computation itself faster.

It can make communication between accelerators faster.

Therefore it improves performance mainly when communication
is a meaningful part of the workload.

## Connection to NPU systems

The transferable concept is not NVLink itself.

Different NPU platforms may use different communication fabrics.

The general reasoning is:

Multiple accelerators
    ↓
communication required
    ↓
communication fabric
    ↓
bandwidth + latency + topology
    ↓
distributed performance

## Module 9 mental model

Data Parallelism:
    "Which request goes to which model replica?"

Tensor Parallelism:
    "How do devices jointly compute a layer?"

Pipeline Parallelism:
    "Which layers execute on which device?"

Communication:
    "What data must move between devices?"

Interconnect:
    "How quickly can that data move?"

Hardware topology:
    "How are the devices physically/logically connected?"

## Interview takeaway

NVLink does not directly make the model compute faster.

It can reduce GPU-to-GPU communication overhead,
which can improve the scaling efficiency of TP/PP
when communication is significant.


<img width="1051" height="767" alt="image" src="https://github.com/user-attachments/assets/93f41130-a0dd-4392-86e6-cc4cf2507978" />
