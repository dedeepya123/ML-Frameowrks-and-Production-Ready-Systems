# Module 9 — Parallelism & Communication
## Lesson 3 — Tensor Parallelism

### Core idea

Tensor Parallelism partitions tensors/computation of a model across
multiple devices.

Unlike Data Parallelism, one model execution can span multiple devices.

### Simple Linear example

    Y = XW

Column parallel:

    W = [W0 | W1]

    GPU0:
        Y0 = XW0

    GPU1:
        Y1 = XW1

    Y = [Y0 | Y1]

Row parallel:

    W = [W0
         W1]

    X = [X0 X1]

    GPU0:
        Y0 = X0W0

    GPU1:
        Y1 = X1W1

    Y = Y0 + Y1

### Data vs Tensor Parallelism

Data Parallelism:

    complete model on each device
    different requests/data per device

Tensor Parallelism:

    model/tensors partitioned across devices
    one request can use multiple devices

### Why?

- Model too large for one device.
- Need more compute for a model execution.
- Reduce per-device model memory.

### What can be partitioned?

- Weight matrices
- Matrix multiplication
- Attention heads
- MLP intermediate dimensions
- Other tensors

Exact partitioning depends on the strategy.

### Communication

Tensor parallelism introduces communication between devices.

Common collective concepts:

- AllGather
- AllReduce
- ReduceScatter
- Broadcast
- Send/Receive

Do not memorize details yet.

Understand:

    distributed computation
          ↓
    partial results
          ↓
    exchange/combine tensors

### System responsibilities

Model architecture:
    Defines the mathematical operations.

Parallelization/model implementation:
    Defines how operations are partitioned.

Inference engine/framework:
    May create/manage distributed workers and execution.

Runtime:
    Executes local operations and may execute communication operations.

Communication layer:
    Implements collective/data movement mechanisms.

Hardware/interconnect:
    Physically moves data between devices.

### Performance model

    Total time
      ≈ computation
        + communication
        + synchronization

More devices do not automatically mean proportional speedup.

### Key mental model

Data Parallelism:

    Request A → GPU0
    Request B → GPU1
    Request C → GPU2

Tensor Parallelism:

    Request A
       ↓
    GPU0 + GPU1 + GPU2
       ↓
    output

### Core distinction

Data Parallelism:
    "Distribute independent work."

Tensor Parallelism:
    "Distribute one piece of work."

### Questions to ask when reading TP code

1. Why is the model being partitioned?
2. What tensor is partitioned?
3. What does each device compute?
4. What data must move?
5. Which collective communication is needed?
6. Which layer implements the partitioning?
7. What interconnect carries the data?
8. Is communication worth the computation saved?
