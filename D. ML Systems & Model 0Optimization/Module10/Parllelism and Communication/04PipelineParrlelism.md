# Lesson 4 — Pipeline Parallelism

## Core idea

Pipeline Parallelism (PP) splits a model across layers/stages and places
different groups of layers on different devices.

Example:

GPU0 → Layers 0–19
GPU1 → Layers 20–39
GPU2 → Layers 40–59
GPU3 → Layers 60–79

Each stage owns the complete parameters of its assigned layers.

## What moves?

Activations move between pipeline stages:

GPU0 → activation → GPU1 → activation → GPU2 → activation → GPU3

Weights remain on the device that owns their layers.

## Difference from Tensor Parallelism

Tensor Parallelism:
- Splits computation/tensors inside a layer.
- One layer can execute across multiple devices.

Pipeline Parallelism:
- Splits layers/stages across devices.
- Each device owns complete layers.

## Why use Pipeline Parallelism?

1. Model does not fit on one device.
2. Reduce per-device model memory.
3. Improve throughput by pipelining multiple microbatches.

## Pipeline bubble

With one request, later stages initially wait for earlier stages.

Multiple microbatches allow stages to overlap:

GPU0: M1 M2 M3 M4
GPU1:    M1 M2 M3 M4
GPU2:       M1 M2 M3 M4
GPU3:          M1 M2 M3 M4

This reduces idle time.

## Communication

PP primarily requires activation transfer between stages.

TP primarily requires communication associated with partitioned tensor computation.

## Combined TP + PP

Large models can use both:

PP:
    split layers into stages

TP:
    split computation inside each stage/layer

## Responsibility

Model:
    Defines the Transformer computation.

Parallelization strategy:
    Decides stage boundaries and tensor partitioning.

Inference/distributed engine:
    Coordinates workers, stages, scheduling and execution.

Runtime:
    Executes local graphs/operators and supported communication.

Interconnect:
    Moves activations/data between devices.

Hardware:
    Performs computation and memory operations.

## Mental model

Data Parallelism:
    "Which request goes to which replica?"

Tensor Parallelism:
    "How do multiple devices jointly compute this layer?"

Pipeline Parallelism:
    "Which layers execute on which device?"

## Key question

Whenever you see Pipeline Parallelism, ask:

"What problem is it solving, what exactly is partitioned,
where are the stage boundaries, what data crosses devices,
and what is the communication/scheduling cost?"
