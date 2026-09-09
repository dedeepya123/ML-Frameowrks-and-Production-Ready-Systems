# Module 10 — Lesson 4: Operator Fusion

## Core idea

**Operator fusion = combining compatible operations in a computation graph into a larger fused computation to improve execution efficiency.**

Example:

```text
Before:
x → Mul → y → Add → z

After:
x → Fused(Mul + Add) → z
```

## Why fuse?

1. **Reduce intermediate memory traffic**

   * Avoid writing an intermediate tensor and reading it again.

2. **Reduce kernel/dispatch overhead**

   * Multiple operations may execute through fewer launches/dispatches.

3. **Improve locality**

   * Intermediate values can remain closer to computation instead of going through slower memory.

## How?

```text
Graph
  ↓
Compiler analyzes dependencies, shapes, dtypes, layouts
  ↓
Recognizes compatible/profitable patterns
  ↓
Creates fused regions
  ↓
Backend lowers/code-generates them
  ↓
Runtime executes
```

## Important distinction

Operator fusion is a **graph/compiler-level optimization**.

It does **not automatically mean exactly one physical hardware kernel**.

```text
Operator Fusion
    ↓
fused computation/region
    ↓
backend/code generation
    ↓
kernel(s) / implementation
```

Kernel fusion will be studied separately.

## Why not fuse everything?

Fusion can hurt when it causes:

* excessive register/local-memory pressure
* very large/complex kernels
* reduced scheduling flexibility
* incompatible layouts/dtypes
* unsupported hardware patterns

Therefore:

> The goal is **profitable fusion**, not maximum fusion.

## Transformer connection

Transformer graphs contain many possible optimization opportunities:

```text
Norm → projections → RoPE → attention → projection
                         ↓
                    residual/add
                         ↓
                       MLP
```

Some operations may be fused depending on the compiler/backend.

FlashAttention is related to the same goal of reducing expensive memory movement, but it is more than simple operator fusion: it combines algorithmic changes, tiling, and online softmax to avoid materializing the full attention matrix.

## Mental model

```text
Graph Capture
      ↓
Graph Representation
      ↓
Global visibility
      ↓
Optimization
      ↓
Operator Fusion
      ↓
Backend / Codegen
      ↓
Kernels
      ↓
Hardware
```

### One-line takeaway

> **Graph capture gives the compiler visibility; operator fusion uses that visibility to combine compatible computation and reduce execution/memory overhead.**
