# Module 6 — Lesson 4: Dequantization & Accumulation

## Accumulation

Quantized inputs may be INT8:

INT8 × INT8
    ↓
products
    ↓
wider accumulator
    ↓
typically INT32

Input dtype and accumulator dtype are different concepts.

---

## Recovering the Real Result

X ≈ sx × Xq
W ≈ sw × Wq

Therefore:

Y ≈ sx × sw × (XqWq)

After accumulation:

INT32 accumulator
       ↓
scale handling
       ↓
output representation

---

## Dequantization

Conceptually:

integer-domain result
       ↓
scale
       ↓
floating-point result

But this does NOT necessarily mean a separate
dequantization kernel exists.

The compiler/kernel may fuse the operation.

---

## Requantization

If the next operation wants a quantized tensor:

INT32
  ↓
scale
  ↓
INT8

This is requantization.

---

## Fusion

Naive:

INT8 GEMM
  ↓
INT32 buffer
  ↓
dequantization
  ↓
FP16 buffer

Optimized:

INT8 GEMM
  ↓
INT32 accumulation
  ↓
scale + conversion
  ↓
FP16 output

Fusion can reduce:
- intermediate buffers
- memory traffic
- kernel launches
- synchronization

---

## Important Distinction

Weight dequantization:

INT4 weight
  ↓
FP16 weight
  ↓
FP16 GEMM

is different from:

Result dequantization:

INT8 × INT8
  ↓
INT32
  ↓
scale
  ↓
FP16

---

## Systems Principle

Mathematical pipeline:

INT8 × INT8
→ INT32
→ scale
→ output

does NOT imply that every step is a separate kernel.

The compiler/runtime decides the physical implementation.

---

## Production Reasoning

When seeing conversion/dequantization code ask:

1. Is it an abstraction or actual operation?
2. Is it lowered by the compiler?
3. Is it fused?
4. Is an intermediate tensor materialized?
5. What dtype does the next operator consume?
6. What does the NPU support?

## Core Principle

Do not confuse:
"What mathematically needs to happen"
with
"How the hardware actually executes it."
