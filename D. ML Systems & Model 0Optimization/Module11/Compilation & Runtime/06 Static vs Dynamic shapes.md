# Module 10 — Lesson 6: Static vs Dynamic Shapes

## Core idea

**Static shape:** tensor dimensions are known/fixed during compilation.

```text
[1, 512, 4096]
```

**Dynamic shape:** one or more dimensions can vary at runtime.

```text
[1, N, 4096]
```

## Why shapes matter

Shape information affects:

* memory allocation
* buffer sizes
* kernel dimensions
* tiling
* scheduling
* fusion opportunities
* memory planning

```text
Known shape
    ↓
more predictable execution
    ↓
more specialization opportunities
```

## Static vs dynamic

| Static                | Dynamic                  |
| --------------------- | ------------------------ |
| Fixed dimensions      | Variable dimensions      |
| Easier planning       | More runtime flexibility |
| Easier specialization | More general execution   |
| Predictable memory    | Harder memory planning   |
| Less flexible         | More flexible            |

Neither is universally better.

## Common strategies

### Fully static

```text
[1, 512, 4096]
```

### Bounded dynamic

```text
N ∈ [1, 2048]
```

### Shape specialization

```text
Compile variants:
128 / 256 / 512 / 1024
```

Runtime selects an appropriate variant.

## Gemma4 / QAIRT connection

Adapted execution uses predictable dimensions such as:

```text
ARN = 521
GCL = 15527

attention_mask:
[1, 1, ARN, GCL]

swa_attention_mask:
[1, 1, ARN, LCL]
```

This is more compiler/runtime-friendly than allowing all dimensions to grow dynamically.

The adaptation therefore helps turn flexible model behavior into a more predictable execution structure.

## Important distinction

Static shape does not mean the conceptual model only supports one input size.

It can mean that a **particular compiled graph/executable is specialized for that shape**.

## Mental model

```text
Graph
  ↓
Operations + dependencies
  ↓
Shapes + dtypes + layouts
  ↓
Compiler
  ├── Fusion
  ├── Kernel generation
  ├── Memory planning
  └── Scheduling
```

### One-line takeaway

> **Static shapes give the compiler stronger knowledge about tensor sizes, making execution, optimization, and memory planning more predictable; dynamic shapes trade some of that predictability for flexibility.**
