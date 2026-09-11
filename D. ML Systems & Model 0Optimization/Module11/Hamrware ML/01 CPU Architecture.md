# Module 11 — Hardware-Aware ML

## Lesson 1 — CPU Architecture for ML

### Why hardware matters

The compiler ultimately has to transform ML computation into an execution strategy for a specific hardware architecture.

```text
Model
 ↓
Graph
 ↓
Compiler
 ↓
Runtime
 ↓
Hardware
```

The same ML operation can execute differently on CPU, GPU, and NPU.

### CPU mental model

A CPU is a **general-purpose processor** built from a relatively small number of powerful, flexible cores.

Each core has computation hardware, including floating-point and vector/SIMD capabilities.

### SIMD

SIMD = **Single Instruction, Multiple Data**

Instead of processing:

```text
x0 → operation
x1 → operation
x2 → operation
x3 → operation
```

a vector instruction can conceptually process:

```text
[x0 x1 x2 x3] → operation → [y0 y1 y2 y3]
```

This is useful for many tensor operations.

### CPU memory hierarchy

```text
Registers
   ↓
L1
   ↓
L2
   ↓
L3
   ↓
DRAM
```

Closer memory is generally smaller and faster.

Performance depends on both:

```text
Compute + Data movement
```

### CPU + ML

For ML, important factors include:

* multiple CPU cores
* SIMD/vectorization
* cache locality
* memory bandwidth
* tiling/data reuse
* compute-bound vs memory-bound behavior

### Compiler connection

The compiler can optimize CPU execution using:

```text
parallelism
+ vectorization
+ tiling
+ memory locality
+ layout
```

### Key mental model

> A CPU provides general-purpose computation with multiple powerful cores, vector execution, and a memory hierarchy. ML performance depends on how effectively the compiler maps tensor computation onto these compute and memory resources.

### What we do NOT need

For ML systems, we do not need deep study of CPU transistor design, ISA encoding, branch-predictor internals, or microarchitecture implementation details.
