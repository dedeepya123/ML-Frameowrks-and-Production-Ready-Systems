# Module 1 — The Inference System

## Lesson 2 — What Actually Happens When One Operator Executes?

## 1. Example

Y = X @ W

If:

X = [M × K]
W = [K × N]

then:

Y = [M × N]

Each output:

Y[i,j] = sum(X[i,k] * W[k,j])

---

## 2. Tensor operation → execution

Conceptually:

Tensor operation
→ framework operator
→ backend/runtime
→ kernel(s)
→ hardware

Important:
An operator does NOT necessarily map 1:1 to one kernel.

Possibilities:

1 operator → 1 kernel

1 operator → multiple kernels

multiple operators → 1 fused kernel

---

## 3. Data movement

Computation requires data.

Conceptually:

Memory
→ load data
→ compute
→ write result

Moving data has a cost.

Therefore performance is not only about arithmetic.

---

## 4. Memory hierarchy

Fast memory is generally smaller.

Slow memory is generally larger.

Typical conceptual hierarchy:

Registers
→ L1
→ L2/cache
→ main/device memory
→ storage

Exact hierarchy depends on CPU/GPU/NPU.

---

## 5. Data reuse

Efficient implementations try to:

load data
→ keep it close to compute
→ reuse it
→ perform many operations
→ write results

Reducing unnecessary memory movement can significantly improve performance.

---

## 6. Compute-bound

An operation is compute-bound when computation is the main limiting factor.

Conceptually:

Compute units are busy
→ arithmetic is limiting performance

---

## 7. Memory-bound

An operation is memory-bound when moving data is the main limiting factor.

Conceptually:

Compute units wait for data
→ memory/bandwidth is limiting performance

---

## 8. FLOPs

FLOP = Floating Point Operation.

For matrix multiplication:

X [M × K] × W [K × N]

approximately:

M × K × N multiplications

and approximately the same number of additions.

Approximate total:

2MKN operations

depending on counting convention.

---

## 9. FLOPs != runtime

Actual performance also depends on:

- memory movement
- memory bandwidth
- kernel efficiency
- cache behavior
- hardware utilization
- synchronization
- kernel launch overhead
- communication
- tensor shapes/layout

Therefore:

Peak FLOPs != actual performance.

---

## 10. Kernel optimization

Two kernels can compute the same mathematical result but have very different performance.

Naive:

load data repeatedly
→ compute
→ poor reuse
→ high memory traffic

Optimized:

load blocks
→ reuse data
→ compute many operations
→ reduce memory traffic
→ write results

---

## 11. Elementwise vs matrix operations

Example:

Y = X + 1

Low arithmetic work
but must read/write the tensor.

Often memory-sensitive.

Example:

Y = X @ W

Large arithmetic workload.

Often compute-intensive, depending on shape and hardware.

These are tendencies, not universal rules.

---

## 12. Kernel launch overhead

If several small operations each become separate kernels:

A → kernel
B → kernel
C → kernel

launch/scheduling overhead can matter.

Fusion can sometimes turn:

A + B + C

into fewer kernels.

---

## 13. Transformer example

Q = X @ Wq

Model view:
Q projection.

Systems view:

X and Wq
→ memory
→ matmul kernel
→ compute hardware
→ Q

Questions:

- What are X and Wq shapes?
- How many FLOPs?
- How many bytes are read?
- How many bytes are written?
- How much data can be reused?
- Is the operation compute-bound?
- Is it memory-bound?
- Which kernel executes it?
- How efficiently is the hardware utilized?

---

## 14. Core mental model

For every operator:

1. What computation is required?
2. What data must be read?
3. Where is that data?
4. What data can be reused?
5. How much computation is performed?
6. How is the work mapped to hardware?
7. What kernel executes it?
8. Is computation or memory the bottleneck?

---

## Key takeaway

Performance is determined by more than the mathematical operation.

We must reason about:

COMPUTATION
+
DATA MOVEMENT
+
MEMORY HIERARCHY
+
KERNEL IMPLEMENTATION
+
HARDWARE UTILIZATION

A mathematically identical operation can have very different performance depending on how it is implemented.
