# Module 10 — Lesson 3: Graph Capture

## Core idea

**Graph capture** is the process of obtaining a computational representation of a model's tensor operations and dependencies.

```text
Python / HF model
       ↓
Capture
       ↓
Graph / IR
```

---

## Why capture?

Python code is flexible, but compilers need a structured representation of computation.

Example:

```python
y = x @ w
z = torch.relu(y)
```

becomes conceptually:

```text
x ───┐
     ▼
   MatMul → y → ReLU → z
     ▲
     │
     w
```

This representation lets the compiler reason about the computation globally.

---

## Conceptual capture process

```text
Model
  ↓
forward(input)
  ↓
tensor operations occur
  ↓
capture mechanism observes/analyzes operations
  ↓
graph nodes are created
  ↓
inputs/outputs/dependencies are connected
  ↓
Graph / IR
```

A graph node can conceptually contain:

```text
operation
inputs
outputs
shape
dtype
attributes
```

---

## Capture is not necessarily tracing

Different systems can construct graphs using different mechanisms:

### Tracing

```text
example inputs
     ↓
execute model
     ↓
record executed tensor operations
     ↓
graph
```

### Program analysis / transformation

```text
model/program
     ↓
analyze/transform
     ↓
graph / IR
```

### Explicit graph construction

```text
graph.add(...)
     ↓
graph
```

Therefore:

> Graph capture does not universally mean "trace one execution."

---

## Important limitation of simple tracing

Consider:

```python
if x.shape[0] > 10:
    y = x + 1
else:
    y = x * 2
```

If captured with a shape where the condition is true, a simple trace may only observe:

```text
x → Add(1) → y
```

The other branch was not executed.

This is why dynamic control flow can make graph capture difficult.

---

## Graph breaks

If part of a model cannot be represented:

```text
Graph 1
   ↓
unsupported/dynamic Python
   ↓
eager execution
   ↓
Graph 2
```

This is called a **graph break**.

Depending on the system, unsupported behavior may instead require adaptation, fallback, specialization, or compilation failure.

---

## Transformer example

A simplified transformer computation:

```text
Input
  ↓
RMSNorm
  ↓
Q/K/V projections
  ↓
RoPE
  ↓
QKᵀ
  ↓
Softmax
  ↓
PV
  ↓
Output projection
  ↓
Residual
  ↓
MLP
  ↓
Output
```

Capture makes these tensor dependencies explicit.

---

## Shapes matter

During capture, the system may know or represent:

```text
shape
dtype
layout
```

For example:

```text
x = [1, S, 4096]
```

where `S` might be:

* fixed/static
* symbolic/dynamic

Shape information helps later with:

* memory planning
* kernel selection
* layout decisions
* hardware constraints
* execution planning

---

## Connection to Gemma4 / QAIRT

Adaptations can make computation more explicit and predictable.

For example:

```text
HF:
model computes mask internally

Adapted:
ModelInputBuilder
      ↓
attention_mask
      ↓
model.forward(attention_mask)
```

The eventual computation can therefore represent the mask as an explicit graph input.

Fixed/constrained shapes such as:

```text
[1, 1, ARN, GCL]
[1, 1, ARN, LCL]
```

can also make memory and execution planning more predictable.

---

## Capture vs compilation

These are different stages.

### Capture

> What computation does the model perform?

```text
Python → Graph
```

### Compilation

> How should this computation execute efficiently on the target?

```text
Graph → target-specific executable
```

Therefore:

```text
Python
  ↓
Capture
  ↓
Graph / IR
  ↓
Compile
  ↓
Runtime
  ↓
Kernels
  ↓
NPU
```

## One-line takeaway

> **Graph capture turns the model's tensor computation into an explicit representation that a compiler can analyze and optimize.**
