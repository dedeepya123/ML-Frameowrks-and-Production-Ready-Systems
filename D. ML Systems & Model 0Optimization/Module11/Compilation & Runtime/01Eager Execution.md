# Module 10 — Compilation & Runtime

## Lesson 1 — Eager Execution

### Core idea

**Eager execution** means tensor operations are executed as the program encounters them rather than first building a complete computation graph.

```text
Python/model code
      ↓
Tensor operation
      ↓
Backend
      ↓
Kernel
      ↓
Hardware
```

### Important distinction

Eager does **not** mean CPU execution.

Python may control execution while the actual tensor operation runs on a GPU/NPU.

```text
Python
  ↓
PyTorch operation
  ↓
Accelerator backend
  ↓
GPU/NPU
```

### Why eager execution is useful

* Easy to write and debug
* Flexible Python control flow
* Dynamic behavior is natural
* Intermediate tensors can be inspected directly

### Limitation

Because operations are executed incrementally, the system may not have a complete view of the computation beforehand.

This can limit opportunities for:

* operator/kernel fusion
* memory planning
* hardware-specific optimization
* reducing intermediate memory traffic
* reducing execution/dispatch overhead

### Eager vs graph execution

Eager:

```text
Op 1 → execute
Op 2 → execute
Op 3 → execute
```

Graph-based:

```text
Build/Capture Graph
        ↓
Optimize
        ↓
Compile/Lower
        ↓
Execute
```

A graph gives the compiler visibility into a larger region of computation.

### Simple example

Eager:

```python
y = x + 1
z = y * 2
```

Conceptually:

```text
x
 ↓
Add
 ↓
temporary y
 ↓
Mul
 ↓
z
```

A compiler may potentially fuse these operations:

```text
x
 ↓
Fused Add + Mul
 ↓
z
```

### Connection to Gemma4 / QAIRT

HF Gemma4 code is flexible and framework-oriented.

Runtime-oriented execution may prefer computation to be more explicit and predictable.

Some adaptation work can therefore move dynamic/preparatory logic outside the model and provide required tensors explicitly.

Example:

```text
HF:
model.forward()
    ↓
construct mask
    ↓
attention

Adapted:
ModelInputBuilder
    ↓
construct mask
    ↓
model.forward(mask)
    ↓
attention
```

This makes the eventual execution representation easier to reason about.

### ML Systems mental model

```text
Model
  ↓
PyTorch/HF representation
  ↓
Eager execution
  ↓
Graph capture
  ↓
Compiler optimization
  ↓
Lowering
  ↓
Runtime
  ↓
Kernel
  ↓
NPU
```

### Key questions

When reading a runtime system, ask:

1. What is the model computation?
2. How is it represented?
3. How is it optimized?
4. How is it lowered?
5. Where does it execute?

### One-line takeaway

> **Eager execution prioritizes flexibility and immediate execution; graph/compiled execution gives the system a global view that enables hardware-oriented optimization.**
