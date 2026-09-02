# Module 9 — Parallelism & Communication
## Lesson 2 — Data Parallelism

### Core idea

Data Parallelism replicates the complete model across multiple devices
and distributes different data/workloads across those replicas.

        Complete Model
        /      |      \
      GPU0   GPU1    GPU2
       R1     R2      R3

### Why?

Primarily to increase throughput/workload capacity.

It is useful when the model already fits on one device.

### What is replicated?

The complete model:

GPU0 → Model
GPU1 → Model
GPU2 → Model

### What is distributed?

Training:
    different batches

Inference:
    different requests/workloads

### Training vs inference

Training:
    different data
        ↓
    gradients
        ↓
    gradient synchronization

Inference:
    different requests
        ↓
    independent execution
        ↓
    outputs

### Important limitation

Data parallelism does NOT combine device memory.

If:

    Model = 140 GB
    GPU = 80 GB

then replicating the model across GPUs does not solve the problem.

The model must fit on each device.

### Who does what?

Serving/orchestration:
    routes requests to workers/devices.

Inference engine:
    manages efficient execution, scheduling/batching,
    KV cache, etc., depending on the engine.

Runtime:
    executes the model on the assigned device.

Kernel:
    implements individual operations.

Hardware:
    performs the actual computation.

### Important distinction

Data Parallelism:
    replicate model + split requests/data

Tensor Parallelism:
    split tensor/computation

Pipeline Parallelism:
    split model layers

### Main tradeoff

Data Parallelism:

    more total model memory
            ↓
    more independent compute capacity
            ↓
    higher throughput

Advantage:
    relatively low cross-device communication for
    independent inference requests.

### Recognition rule

If every device has the complete model and different
requests/data are assigned to different devices:

    → Data Parallelism / model replication
