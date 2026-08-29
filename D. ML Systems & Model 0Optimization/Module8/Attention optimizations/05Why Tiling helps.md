# Module 7 — Lesson 5: Why Tiling Helps

## Definition

Tiling is the technique of dividing a large computation into
smaller blocks/tiles that fit into a faster local memory level.

The goal is to improve locality and data reuse while reducing
expensive memory traffic.

---

## Core Idea

Instead of:

large data
   ↓
repeated memory access
   ↓
compute

Use:

large data
   ↓
load small tile
   ↓
keep tile in fast/local memory
   ↓
reuse tile multiple times
   ↓
compute
   ↓
move to next tile

---

## Why It Helps

Tiling provides:

- better data locality
- more data reuse
- less off-chip memory traffic
- better use of local/on-chip memory
- higher effective arithmetic intensity
- better compute utilization

---

## Matrix Multiplication Example

C = A × B

Each output:

C[i,j] = sum_k A[i,k] × B[k,j]

The same A/B values participate in many output calculations.

Therefore:

load a tile once
      ↓
reuse it for multiple calculations
      ↓
reduce repeated memory movement

---

## Memory Hierarchy

Conceptually:

Registers
   ↑
Local SRAM / shared memory
   ↑
Cache
   ↑
DRAM / HBM
   ↓
slower and generally farther from compute

Tiling attempts to keep the active working set in a
faster memory level.

---

## Connection to Arithmetic Intensity

AI = FLOPs / Bytes moved

Tiling does not primarily reduce FLOPs.

Instead it attempts to reduce unnecessary bytes moved
and increase computation performed per loaded byte.

Therefore:

data reuse ↑
memory traffic ↓
effective AI ↑

---

## Tile Size Tradeoff

Tile too small:
- poor reuse
- more tile loads
- more memory overhead

Tile too large:
- may not fit local memory
- may reduce parallelism
- may increase resource pressure

Good tile:
- fits local memory
- provides useful reuse
- keeps compute units busy

Tile size depends on hardware.

---

## Hardware Dependence

CPU:
- cache blocking
- SIMD/vectorization

GPU:
- shared-memory tiling
- register tiling
- tensor cores

NPU:
- local SRAM tiling
- matrix engines
- DMA/data movement

The mathematical operation can be identical while the
best tiling strategy differs by hardware.

---

## Connection to FlashAttention

FlashAttention applies the same principle to attention.

Instead of materializing the entire:

QKᵀ = [N × N]

matrix:

1. Divide Q/K/V into tiles
2. Load useful tiles into fast local memory
3. Compute score tile
4. Apply online softmax processing
5. Multiply with V tile
6. Update running output
7. Discard temporary intermediates
8. Process next tile

Therefore FlashAttention benefits from:
- tiling
- locality
- data reuse
- reduced off-chip memory traffic

---

## Model → Hardware Reasoning

Model:
    Attention

Math:
    softmax(QKᵀ)V

Algorithm:
    tiled/blockwise attention

Kernel:
    load → compute → accumulate → next tile

Memory:
    external memory → local memory → registers/compute

Hardware:
    CPU / GPU / NPU

---

## Best Explanation

"Tiling divides a large computation into smaller blocks that
fit into faster local memory. We load a block once and reuse
it for multiple computations instead of repeatedly fetching
the same data from slower memory. This improves locality,
reduces memory traffic, and increases effective arithmetic
intensity while performing essentially the same computation."

---

## Core Mental Model

Tiling
   ↓
small working set
   ↓
fits in fast memory
   ↓
data stays close to compute
   ↓
reuse data
   ↓
less memory movement
   ↓
better performance
