Lesson 5 — Communication Cost

So far we've said:

Data Parallelism → distribute requests
Tensor Parallelism → split computation inside layers
Pipeline Parallelism → split layers across devices

Now comes the unavoidable question:

If we split work across devices, how do those devices exchange the data they need?

And more importantly:

When does communication become expensive enough that adding more devices actually makes things slower?

That's the heart of this lesson.

1. The fundamental problem

Imagine one accelerator:

Input
  ↓
GPU
  ↓
Output

Everything is local.

Now use two GPUs:

        ┌───── GPU0
Input ──┤
        └───── GPU1

The moment GPU0 needs something from GPU1, data has to physically move.

So distributed execution becomes:

Compute
   ↓
Communication
   ↓
Compute
   ↓
Communication
   ↓
Compute

Instead of simply:

Compute → Compute → Compute

That communication has:

latency
bandwidth limits
synchronization cost
memory movement cost
2. A very simple analogy

Imagine two people solving a problem.

One person
Person A
  ↓
does everything

No communication required.

Two people
Person A ←──── information ────→ Person B

Now they can potentially finish faster.

But every time A says:

"I need your result."

B has to communicate.

If they spend more time talking than solving:

Communication > computation benefit

adding the second person doesn't help.

That's exactly what happens with accelerators.

3. Communication has two major dimensions

A very useful model is:

Communication cost ≈ latency + data-transfer time

More formally:

$$ T_{comm} \approx \alpha + \frac{S}{B} $$

where:

\(S\) = amount of data transferred
\(B\) = effective bandwidth
\(\alpha\) = communication startup/latency cost

This simple equation will appear again and again in distributed ML systems.

4. What does bandwidth mean?

Suppose an interconnect can move:

100 GB/s

and you need to transfer:

10 GB

Ignoring overhead:

$$ T = \frac{10GB}{100GB/s} $$ $$ =0.1s $$

So the transfer itself takes roughly:

100 ms

Now imagine your computation only takes:

20 ms

You have a problem.

Compute       = 20 ms
Communication = 100 ms

Your distributed system is dominated by communication.

5. Why latency matters separately

Now imagine transferring only:

1 KB

The amount of data is tiny.

But suppose establishing the communication operation costs:

50 μs

Then even though the data itself is tiny, you still pay that startup cost.

That's why:

Many small communications can be worse than fewer large communications.

This becomes especially important for fine-grained distributed computation.

6. Now connect this to Tensor Parallelism

Recall our TP example:

$$ Y = XW $$

Suppose:

W = [W0 | W1]

Then:

GPU0:
Y0 = XW0

GPU1:
Y1 = XW1

and:

Y = [Y0 | Y1]

Potentially no reduction is required for this particular column-parallel operation.

But consider row parallelism:

$$ Y = X_0W_0 + X_1W_1 $$

GPU0 computes:

Y0 = X0W0

GPU1 computes:

Y1 = X1W1

Now we need:

Y = Y0 + Y1

So:

GPU0 ── Y0 ──┐
             ├── reduction
GPU1 ── Y1 ──┘

Communication is now part of the computation.

7. This is where collectives appear

Distributed ML uses common communication patterns called collective operations.

You don't need to memorize all of them yet. Understand the concepts.

All-Reduce

Everyone contributes data and everyone gets the combined result.

GPU0: A ──┐
GPU1: B ──┼──> A+B ──> everyone
GPU2: C ──┘

Result:

GPU0 → A+B+C
GPU1 → A+B+C
GPU2 → A+B+C

Very common in distributed training and can also appear in model-parallel execution.

All-Gather

Each device has part of the data.

Everyone needs the complete data.

GPU0: A
GPU1: B
GPU2: C

After all-gather:

GPU0: A B C
GPU1: A B C
GPU2: A B C
Reduce-Scatter

The opposite-ish combination:

Each device contributes data, but instead of everyone receiving the complete result, each gets one portion of the reduced result.

Conceptually:

GPU0 ─┐
GPU1 ─┼── reduce ──> split result
GPU2 ─┘

Useful for distributed tensor computations.

Point-to-Point

One device sends directly to another:

GPU0 ─────────> GPU1

This is especially intuitive for Pipeline Parallelism.

GPU0 finishes its stage:

activation
    ↓
GPU1
8. Communication patterns map nicely to our parallelism

Now connect everything:

Parallelism	Typical communication
Data Parallelism	synchronization between replicas, especially training
Tensor Parallelism	all-reduce / all-gather / reduce-scatter / etc.
Pipeline Parallelism	point-to-point activation transfer

The exact operations depend on the implementation, but this is the useful mental model.

9. Pipeline Parallelism communication

Suppose:

GPU0 → Layers 0–19
GPU1 → Layers 20–39

GPU0 produces:

X = [batch, seq, hidden]

It must send X to GPU1.

So:

GPU0
  │
  │ activation
  ▼
GPU1

If the activation is large, communication can become expensive.

For example:

Activation = 100 MB
Interconnect = 100 GB/s

Transfer time:

$$ 100MB / 100GB/s \approx 1ms $$

If the stage itself takes:

20 ms

that's relatively manageable.

But if the stage takes:

2 ms

then communication is a much larger fraction of the execution time.

10. This gives us an important ratio

Suppose:

Compute = 2 ms
Communication = 1 ms

Then:

Total ≈ 3 ms

Communication is:

$$ \frac{1}{3} \approx 33\% $$

But suppose:

Compute = 20 ms
Communication = 1 ms

Then:

Total ≈ 21 ms

Communication is only:

$$ \frac{1}{21} \approx 4.8\% $$

So the exact same interconnect can be:

insignificant for one workload
a major bottleneck for another

This is why communication must always be considered relative to computation.

11. Why faster hardware doesn't automatically solve it

Suppose we have:

GPU0 compute = 20 ms
GPU1 compute = 20 ms

Single GPU:

40 ms

Two GPUs:

GPU0 = 20 ms
GPU1 = 20 ms

You might think:

20 ms

Great — 2× speedup.

But now communication costs:

10 ms

Total:

20 + 10 = 30 ms

Speedup:

$$ 40/30 = 1.33\times $$

Not 2×.

And if communication were:

30 ms

then:

20 + 30 = 50 ms

You've actually made it slower than one GPU.

This is one of the most important principles in distributed ML:

Parallelism gives you more compute resources, but communication determines how effectively you can use them.

12. Communication and synchronization

There's another hidden cost.

Imagine:

GPU0 → finishes in 5 ms
GPU1 → finishes in 8 ms
GPU2 → finishes in 20 ms
GPU3 → finishes in 7 ms

If the next operation requires all four:

GPU0 ── done
GPU1 ── done
GPU2 ── still working
GPU3 ── done
             ↓
        synchronization
             ↓
         continue

GPU0, GPU1 and GPU3 may sit idle waiting for GPU2.

So distributed execution has two different problems:

Communication

Moving data.

Synchronization

Waiting for other devices.

Both reduce utilization.

13. This is why load balancing matters

Suppose four pipeline stages take:

Stage 0 = 10 ms
Stage 1 = 10 ms
Stage 2 = 40 ms
Stage 3 = 10 ms

The pipeline cannot run at 10 ms per stage.

Stage 2 becomes the bottleneck.

GPU0   10ms
GPU1   10ms
GPU2   40ms  ← bottleneck
GPU3   10ms

So PP needs reasonably balanced stages.

Instead, we'd prefer something closer to:

GPU0   20ms
GPU1   20ms
GPU2   20ms
GPU3   20ms

This is another systems-level tradeoff:

Memory balance + compute balance + communication cost must all be considered when partitioning a model.

14. Now let's connect PCIe and NVLink

You already have PCIe coming next in Lesson 6.

For now, just establish the intuition.

Suppose:

GPU0 ── PCIe ── GPU1

versus:

GPU0 ── high-speed interconnect ── GPU1

The faster interconnect provides greater effective communication capability and/or lower latency.

Therefore:

Same model
Same parallelism
Same GPUs
Different interconnect
        ↓
Different performance

This is why the system topology matters.

15. Topology matters too

Imagine four GPUs:

GPU0 ─ GPU1
  │
  │
GPU2 ─ GPU3

Some GPU pairs may have a faster connection than others.

Therefore communication:

GPU0 → GPU1

might be cheaper than:

GPU0 → GPU3

A distributed runtime/inference engine can therefore care about:

Which device communicates with which device?

This becomes increasingly important as we move toward PCIe, NVLink, NUMA, and distributed accelerator systems.

16. Now the bigger picture

We can now expand our execution model.

Previously:

Model
 ↓
Runtime
 ↓
Kernel
 ↓
Hardware

With distributed execution:

             Model
               ↓
      Parallelization Strategy
               ↓
       ┌───────┴────────┐
       ↓                ↓
    Device 0          Device 1
       │                │
     Runtime          Runtime
       │                │
     Kernel           Kernel
       │                │
    Hardware         Hardware
       │                │
       └──── Communication ────┘

And communication itself has its own stack:

Distributed framework / engine
            ↓
Communication library/runtime
            ↓
Interconnect
            ↓
Device memory

This is why ML Systems is not just model computation.

It's also:

Where does the data go?

17. Your "why / when / how / who / where" framework

Let's apply it.

Why communication?

Because computation has been distributed across devices and they need to exchange data.

When?

Whenever a distributed computation requires data from another device.

How?

Through communication primitives such as:

point-to-point
all-reduce
all-gather
reduce-scatter
Who decides?

The distributed execution / parallelization strategy determines what communication is required.

The framework/engine and communication stack execute it.

Where?

Across device memory and an interconnect such as PCIe, NVLink, or another accelerator interconnect.

What is the cost?

Approximately:

$$ T_{comm} \approx \alpha + \frac{S}{B} $$

plus synchronization and implementation overhead.

18. The most important systems equation so far

When we parallelize:

$$ T_{total} \neq \frac{T_{compute}}{N} $$

in general.

Instead, think:

$$ T_{total} \approx T_{compute} + T_{communication} + T_{synchronization} $$

And sometimes:

$$ T_{communication} $$

becomes the dominant term.

That's the exact reason why:

Scaling from 1 → 2 → 4 → 8 → 16 accelerators doesn't guarantee linear speedup.

19. And this connects beautifully to your earlier learning

Remember Module 2?

We discussed:

Compute-bound
vs
Memory-bound

Now we're adding another dimension:

Compute
   │
   ├── Compute-bound
   │
   ├── Memory-bound
   │
   └── Communication-bound

A distributed workload can become communication-bound.

For example:

Accelerator compute capability ↑↑
        ↓
computation becomes faster
        ↓
communication stays similar
        ↓
communication becomes larger fraction
        ↓
scaling efficiency falls

This is a very important ML Systems pattern.

20. The mental model I want you to leave with

Don't think:

"I have 4 GPUs, therefore I have 4× compute."

Think:

"I have 4 compute resources connected by finite-bandwidth communication links."

Then ask:

How much computation?
How much data must move?
How often must it move?
How fast is the interconnect?
Do devices need to synchronize?
Are stages balanced?
Can communication overlap with computation?

Those questions let you reason about real distributed performance, rather than just counting accelerators.
