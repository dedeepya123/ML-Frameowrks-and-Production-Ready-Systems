# Module 9 — Parallelism & Communication
## Lesson 1 — Why Parallelize?

### Core idea

Parallelism distributes model computation, model memory, or workload across
multiple accelerators.

### Why multiple accelerators?

1. Model does not fit in one device's memory.
2. One device does not provide enough compute / latency.
3. One device cannot handle required request throughput.

### Important distinction

Multiple devices can be used by:

- Replicating the model and distributing requests.
- Splitting computation within layers.
- Splitting layers across devices.

These lead to:

- Data Parallelism
- Tensor Parallelism
- Pipeline Parallelism

### Key systems insight

Multiple devices introduce communication.

Therefore:

More devices ≠ automatically faster.

End-to-end performance depends on:

    computation + communication + synchronization

### Reasoning framework

When seeing a distributed ML system, ask:

1. Why are multiple devices needed?
2. What is being distributed?
3. What data must move between devices?
4. How expensive is the communication?
5. Does the parallelism provide a net performance benefit?

### Mental model

Model / Workload
      ↓
Why parallelize?
      ↓
Data / Tensor / Pipeline Parallelism
      ↓
Communication
      ↓
Interconnect
      ↓
Compute + Communication
      ↓
End-to-end performance
