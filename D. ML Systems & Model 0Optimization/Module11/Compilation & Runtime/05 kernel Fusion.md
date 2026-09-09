# Module 10 — Lesson 5: Kernel Fusion

## Core idea

**Kernel fusion = implementing multiple compatible computations together in a single kernel/fused device execution unit so intermediate results do not need to be unnecessarily materialized between separate kernels.**

Example:

```text
Before:

x → Kernel(Mul) → memory → Kernel(Add) → output


After:

x → Kernel(Mul + Add) → output
```

## Main benefits

### 1. Reduce memory traffic

Avoid:

```text
compute → write intermediate → read intermediate → compute
```

Instead:

```text
compute → directly continue → write final result
```

### 2. Reduce kernel/dispatch overhead

```text
Many small kernels
      ↓
Fewer larger kernels
```

### 3. Improve locality

Intermediate values may remain in registers or other faster local storage instead of going through global memory.

## Operator Fusion vs Kernel Fusion

**Operator fusion:**

```text
Graph/compiler level
A → B → C
     ↓
identify a fusible region
```

**Kernel fusion:**

```text
Backend/code-generation level
fused region
     ↓
generate efficient device implementation
```

Operator fusion does **not always guarantee one physical kernel**.

## Why not fuse everything?

Excessive fusion can cause:

* register pressure
* large kernels
* resource pressure
* lower occupancy
* reduced scheduling flexibility
* unsupported hardware patterns

Therefore:

> **The goal is profitable fusion, not maximum fusion.**

## Connection to memory optimization

Kernel fusion directly connects to:

* memory bandwidth
* intermediate tensor materialization
* arithmetic intensity
* roofline reasoning

It is particularly valuable when unnecessary memory traffic dominates execution.

## Connection to FlashAttention

FlashAttention follows the broader principle of reducing expensive intermediate memory traffic, but it is more than simple kernel fusion.

It combines:

* tiling
* online softmax
* on-chip/local data reuse
* an algorithmic execution strategy

to avoid materializing the full attention matrix in HBM.

## Connection to QAIRT/model adaptation

Model adaptation is upstream of these optimizations:

```text
HF Gemma4
    ↓
Adaptation
    ↓
Graph / IR
    ↓
Compiler optimization
    ↓
Operator fusion
    ↓
Kernel generation/fusion
    ↓
Runtime
    ↓
NPU
```

The adaptation makes the model computation and inputs suitable for the target execution/compilation system; the compiler/backend can then perform target-specific optimizations.

## Mental model

```text
Model
  ↓
Graph Capture
  ↓
Graph / IR
  ↓
Operator Fusion
  ↓
Fused Computation
  ↓
Kernel Fusion / Code Generation
  ↓
Runtime
  ↓
Hardware
```

### One-line takeaway

> **Operator fusion decides what computation can be combined; kernel fusion is about realizing that combined computation efficiently at the device-execution level.**
