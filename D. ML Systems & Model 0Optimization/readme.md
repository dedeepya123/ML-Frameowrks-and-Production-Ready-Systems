``` text
MODULE 1  → Inference System & Bottlenecks
```
``` text
Module 2 — Compute vs Memory
- Lesson 1 — FLOPs and Computational Cost
- Lesson 2 — Memory Footprint
- Lesson 3 — Memory Bandwidth
- Lesson 4 — Compute-bound vs Memory-bound
- Lesson 5 — Arithmetic Intensity
- Lesson 6 — Roofline-style Reasoning
MODULE 3  → Transformer Inference Cost

Lesson 1 — Cost of Embeddings
Lesson 2 — Q/K/V Projections
Lesson 3 — Attention Computation
Lesson 4 — MLP Computation
Lesson 5 — LM Head
Lesson 6 — Full Transformer Layer Cost
Lesson 7 — Prefill vs Decode Cost

MODULE 4  → KV Cache Systems
Lesson 1 — KV Cache Memory Calculation
Lesson 2 — How KV Cache Grows
Lesson 3 — Memory Bandwidth During Decode
Lesson 4 — Batch Size vs KV Cache
Lesson 5 — Context Length vs KV Memory
Lesson 6 — MHA vs MQA vs GQA
Lesson 7 — KV Cache Optimizations

MODULE 5  → Quantization Fundamentals
Lesson 1 — Why Lower Precision?
Lesson 2 — FP32 / FP16 / BF16
Lesson 3 — INT8 / INT4
Lesson 4 — Quantization Mathematics
Lesson 5 — Scale and Zero Point
Lesson 6 — Per-tensor vs Per-channel
Lesson 7 — Weight vs Activation Quantization
Lesson 8 — Quantization Error

MODULE 6  → Quantization Implementation

Lesson 1 — What Happens to a Linear Layer?
Lesson 2 — Quantized Weight Representation
Lesson 3 — Quantized Matrix Multiplication
Lesson 4 — Dequantization / Accumulation
Lesson 5 — Quantized Activations
Lesson 6 — Quantized KV Cache
Lesson 7 — Reading Quantization APIs and Production Code

MODULE 7  → Attention Optimizations
Lesson 1 — Naive Attention

We'll first implement/reason about the straightforward version.

Lesson 2 — Where Naive Attention Wastes Memory
Lesson 3 — Memory-efficient Attention
Lesson 4 — FlashAttention
Lesson 5 — Why Tiling Helps
Lesson 6 — Paged Attention
Lesson 7 — Production Attention Implementations

The key question throughout:

What bottleneck is this optimization removing?

MODULE 8  → Batching & Scheduling
Lesson 1 — Why Batching?
Lesson 2 — Static Batching
Lesson 3 — Dynamic Batching
Lesson 4 — Continuous Batching
Lesson 5 — Prefill/Decode Scheduling
Lesson 6 — KV Cache Management
Lesson 7 — Scheduler → Runtime → Model Flow

This is where generate() knowledge starts connecting to actual serving systems.

MODULE 9  → Parallelism & Communication

Lesson 1 — Why Parallelize?
Lesson 2 — Data Parallelism
Lesson 3 — Tensor Parallelism
Lesson 4 — Pipeline Parallelism
Lesson 5 — Communication Cost
Lesson 6 — PCIe
Lesson 7 — NVLink / High-speed Interconnects
Lesson 8 — Compute vs Communication Tradeoffs

MODULE 10 → Compilation & Runtime

Lesson 1 — Eager Execution
Lesson 2 — Graph Representation
Lesson 3 — Graph Capture
Lesson 4 — Operator Fusion
Lesson 5 — Kernel Fusion
Lesson 6 — Static vs Dynamic Shapes
Lesson 7 — Memory Planning
Lesson 8 — Compilation and Runtime Execution
Lesson 9 — Reading Runtime/Compiler APIs

MODULE 11 → Hardware-Aware ML
Lesson 1 — CPU Architecture for ML
Lesson 2 — GPU Architecture
Lesson 3 — NPU Architecture
Lesson 4 — SIMD / Vectorization
Lesson 5 — Tensor/Matrix Compute Units
Lesson 6 — On-chip vs Off-chip Memory
Lesson 7 — Memory Hierarchy
Lesson 8 — Why the Same Model Performs Differently on Different Hardware

MODULE 12 → Edge / On-Device Inference

Lesson 1 — Constraints of Edge Devices
Lesson 2 — Model Size
Lesson 3 — Memory Constraints
Lesson 4 — Latency Constraints
Lesson 5 — Power/Energy
Lesson 6 — Quantization + Hardware
Lesson 7 — Operator Support
Lesson 8 — Model Adaptation for a Runtime
Lesson 9 — CPU/GPU/NPU Execution Decisions

MODULE 13 → End-to-End Optimization

Lesson 1 — Establish a Baseline
Lesson 2 — Profile
Lesson 3 — Find the Real Bottleneck
Lesson 4 — Form an Optimization Hypothesis
Lesson 5 — Implement the Optimization
Lesson 6 — Benchmark
Lesson 7 — Accuracy Validation
Lesson 8 — Regression Testing
Lesson 9 — Production Tradeoffs
Lesson 10 — End-to-End Case Study

```
