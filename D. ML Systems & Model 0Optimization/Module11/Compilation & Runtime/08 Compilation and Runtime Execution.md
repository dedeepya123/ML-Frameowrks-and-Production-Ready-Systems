# Module 10 — Lesson 8: Compilation & Runtime Execution

## Core Idea

**Compiler prepares computation; runtime executes it.**

```text
PyTorch / HF
    ↓
Eager execution
    ↓
Graph capture
    ↓
Graph / IR
    ↓
Optimization
    ├── Fusion
    ├── Shape specialization
    └── Memory planning
    ↓
Lowering
    ↓
Code generation
    ↓
Compiled artifact
    ↓
Runtime
    ↓
Kernel launch
    ↓
Hardware / NPU
```

## Compiler

The compiler answers:

> How should this computation be implemented efficiently for the target hardware?

Typical stages:

```text
Graph
 ↓
Optimization
 ↓
Lowering
 ↓
Code generation
 ↓
Executable / kernels / artifact
```

Compiler may perform:

* operator/kernel fusion
* shape specialization
* layout optimization
* memory planning
* target-specific lowering
* kernel generation

## Runtime

The runtime answers:

> How do I execute the compiled computation now?

Typical responsibilities:

* load compiled artifact
* initialize execution environment
* allocate/manage buffers
* bind inputs and outputs
* manage persistent state
* launch kernels
* synchronize execution
* return outputs

## Compilation vs Runtime

| Compilation            | Runtime                  |
| ---------------------- | ------------------------ |
| Usually infrequent     | Repeated frequently      |
| Transforms graph       | Executes compiled result |
| Optimizes              | Manages execution        |
| Generates target code  | Launches target work     |
| Decides implementation | Manages buffers/state    |

Mental model:

```text
Compile once
     ↓
Execute many times
```

## Lowering

Lowering transforms high-level operations toward target-specific execution:

```text
High-level Graph
      ↓
Compiler IR
      ↓
Target-specific representation
      ↓
Kernels / executable
```

## Runtime and KV Cache

LLM inference involves persistent state:

```text
Input
  ↓
Graph execution
  ↓
Output
  ↓
KV Cache
  ↓
Next inference step
```

Therefore runtime execution is not always simply:

```text
input → output
```

It may also manage state that survives across invocations.

## QAIRT Connection

Conceptually:

```text
Gemma4 HF
   ↓
Adaptation
   ↓
QAIRT-compatible representation
   ↓
Compilation
   ↓
Compiled artifact
   ↓
QAIRT Runtime
   ↓
NPU
```

Adaptation prepares the model for downstream execution.
The compiler creates the target-specific executable representation.
The runtime repeatedly executes it.

## Four Questions to Always Ask

1. **What computation?** → Graph / IR
2. **How optimize it?** → Compiler
3. **How implement it for hardware?** → Lowering + code generation
4. **How execute it repeatedly?** → Runtime

## Key Distinction

> **Adaptation is not compilation, and compilation is not runtime execution.**

They are different stages of the same pipeline.
