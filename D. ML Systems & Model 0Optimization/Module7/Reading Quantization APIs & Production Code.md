# Module 6 — Lesson 7: Reading Quantization APIs & Production Code

## Core Goal

Do not memorize quantization APIs.

Learn to trace:

Model
→ Framework
→ Quantized representation
→ Operator
→ Runtime
→ Kernel
→ Hardware

---

## HF / Framework Level

HF describes model semantics.

Example:

Gemma
→ DecoderLayer
→ Attention
→ q_proj
→ Linear

This tells us WHAT computation the model wants.

It does not necessarily tell us HOW the NPU executes it.

---

## Quantization Transformation

Possible designs:

1. Replace Linear with QuantLinear
2. Replace/store quantized parameters
3. Keep Python model mostly unchanged and quantize during conversion
4. Let compiler/runtime introduce quantization

Therefore:

Python class ≠ guaranteed physical implementation.

---

## Read Quantization Code By Following Data

Ask:

1. What is being quantized?
2. What is its representation before?
3. What is its representation after?
4. Where are scales stored?
5. What is the quantization granularity?
6. Is the data packed?
7. Who consumes the representation?
8. What operator is produced?
9. What kernel executes it?
10. What does the NPU actually receive?

---

## Framework vs Runtime

Framework:

"Compute Linear(x)."

Runtime:

"How can this computation be represented and executed efficiently on this target?"

Pipeline:

HF/PyTorch
    ↓
framework operator
    ↓
conversion / graph
    ↓
runtime representation
    ↓
kernel
    ↓
NPU

---

## Important Principle

High-level model semantics
≠
physical accelerator implementation.

The same model can execute differently on:

CPU
GPU
NPU

---

## Debugging / Code Reading

Read from high level downward:

Model
→ Module
→ Tensor
→ Quantization
→ Operator
→ Runtime
→ Kernel
→ Hardware

At every level ask:

"What is the data representation?"

---

## Core Mental Model

WHAT does the model want?
        ↓
Framework API
        ↓
What representation?
        ↓
Runtime operator
        ↓
What kernel?
        ↓
What hardware?
        ↓
What bottleneck?

This is the model-level → systems-level → compute-level →
memory-level reasoning we are building.
