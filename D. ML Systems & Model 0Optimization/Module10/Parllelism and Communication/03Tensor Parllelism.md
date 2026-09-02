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

Whenever you encounter "Tensor Parallelism" in a paper, codebase, or inference engine, ask:

1. Why?
Does the model not fit?
Is computation too expensive?
Is latency too high?
2. What is partitioned?
Weight columns?
Weight rows?
Attention heads?
MLP intermediate dimension?
Something else?
3. What does each device compute?

Write the actual equation.

For example:

Y0 = XW0
Y1 = XW1
4. What needs to be communicated?
X?
Y?
Partial sums?
Attention outputs?
5. Which collective?
AllGather?
AllReduce?
ReduceScatter?
6. Who implements it?
Model code?
Distributed framework?
Inference engine?
Communication library?
Runtime?
7. How does the data physically move?
``` text
GPU memory
   ↓
communication mechanism
   ↓
PCIe / NVLink / interconnect
   ↓
other GPU
8. Is communication cheaper than the computation we saved?

That's ultimately the performance question.
8. What interconnect carries the data?
9. Is communication worth the computation saved?
```
<img width="1217" height="672" alt="image" src="https://github.com/user-attachments/assets/e84c58df-f815-40dc-9e92-a38bd5913d38" />
