# Module 10 — Lesson 2: Graph Representation

## Core idea

A computation graph represents:

> **Operations + data dependencies**

```text
Input
  ↓
Operation
  ↓
Operation
  ↓
Output
```

Nodes represent operations such as:

```text
MatMul
Add
Softmax
RMSNorm
Reshape
```

Edges represent tensor/data dependencies.

---

## Why do we need graphs?

Eager execution processes operations incrementally.

A graph gives the compiler a broader/global view of the computation.

This enables opportunities for:

* operator/kernel fusion
* memory planning
* scheduling
* parallel execution
* layout optimization
* hardware-specific lowering
* eliminating unnecessary work

---

## Python code → graph

Example:

```python
def forward(x, w):
    y = x @ w
    z = torch.relu(y)
    return z
```

Conceptually:

```text
x ───┐
     ├── MatMul ──→ y ──→ ReLU ──→ z
w ───┘
```

The graph represents the tensor computation, not the Python source code itself.

---

## Conceptual capture process

```text
Python/HF model
      ↓
execute / capture / analyze
      ↓
identify tensor operations
      ↓
record dependencies
      ↓
construct graph / IR
```

Graph capture is therefore the process of discovering/building the computational representation.

Graph execution is different:

```text
Graph
  ↓
Compiler / lowering
  ↓
Executable representation
  ↓
Runtime
  ↓
Kernels
  ↓
Hardware
```

---

## What graph information can contain

A node can conceptually describe:

```text
Operation
Inputs
Outputs
Shapes
Dtypes
Attributes
Constants
```

The exact representation depends on the framework/compiler.

---

## Graph vs Python program

Python program contains:

```text
variables
functions
classes
loops
conditionals
assignments
```

Graph focuses on:

```text
tensor values
operations
dependencies
shapes
dtypes
constants
outputs
```

Therefore:

> Python describes the computation; the graph represents the computation.

---

## Graph vs IR

A graph is a conceptual representation of operations and dependencies.

**IR (Intermediate Representation)** is a broader compiler representation that can contain:

* operations
* types
* shapes
* layouts
* attributes
* control flow
* memory information
* other compiler metadata

So:

```text
Graph ≠ necessarily the entire IR
```

---

## Connection to Gemma4 / QAIRT

HF Gemma4 contains flexible Python/model logic.

Runtime-oriented execution may prefer explicit, predictable tensor computation.

Adaptations such as moving mask construction outside the model can conceptually turn:

```text
dynamic model-side behavior
```

into:

```text
explicit tensor inputs
```

For example:

```text
ModelInputBuilder
     ↓
attention_mask
swa_attention_mask
cache-related inputs
     ↓
Gemma attention computation
```

This can make the computation easier to represent in a constrained execution environment.

---

## ML Systems mental model

```text
Python / HF model
        ↓
Tensor operations
        ↓
Graph / IR
        ↓
Compiler
        ↓
Runtime
        ↓
Kernels
        ↓
NPU
```

### Key question

When reading runtime/compiler code, ask:

> **What computation is being represented, what are its dependencies, and what information does the compiler know at this stage?**

### One-line takeaway

> **A computation graph turns tensor computation into an explicit representation of operations and dependencies, giving the compiler a global view from which it can optimize and lower the workload for hardware.**
