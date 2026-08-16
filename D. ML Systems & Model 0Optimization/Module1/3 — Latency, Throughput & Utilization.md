# Module 1 — The Inference System

## Lesson 3 — Latency, Throughput & Utilization

## 1. Latency

Latency = time taken by one request/inference.

Example:

One inference = 100 ms

Latency = 100 ms.

Latency matters strongly for interactive applications.

---

## 2. Throughput

Throughput = amount of work completed per unit time.

Examples:

- requests/sec
- tokens/sec
- images/sec
- samples/sec

Example:

100 requests in 10 seconds

= 10 requests/sec.

---

## 3. Latency != Throughput

A request can take 100 ms individually while a system processes many requests concurrently.

Batching can increase throughput even if individual-request latency increases.

Example:

Batch 1:
100 ms/request
≈ 10 req/sec sequentially

Batch 8:
8 requests in 150 ms
≈ 53 req/sec

Therefore:

Higher throughput does not necessarily mean lower individual latency.

---

## 4. Batching

Batching combines multiple requests/examples into one execution.

Small batch:

- often lower latency
- less parallel work
- potentially lower throughput

Large batch:

- more parallel work
- potentially better hardware utilization
- potentially higher throughput
- may increase individual latency

Exact behavior depends on hardware and workload.

---

## 5. Utilization

Utilization measures how effectively hardware resources are being used.

Example:

GPU capacity = 100%

Actual useful utilization = 20%

The GPU may be underutilized.

Low utilization can result from:

- small workloads
- insufficient batching
- memory stalls
- synchronization
- kernel launch overhead
- poor tensor shapes
- insufficient parallelism

High utilization does not automatically mean good user latency.

---

## 6. LLM-specific metrics

### TTFT

Time To First Token.

Conceptually:

request
→ prefill
→ first generated token

TTFT is strongly related to prefill performance.

### Decode speed

Often measured as tokens/sec.

Example:

20 tokens/sec

≈ 50 ms/token.

Decode performance is important for long generated outputs.

---

## 7. Prefill vs Decode metrics

Prefill:

prompt
→ process input
→ populate KV cache
→ first token

Related metric:

TTFT

Decode:

one/few new tokens
→ reuse KV cache
→ generate next token
→ repeat

Related metric:

tokens/sec

---

## 8. Total generation time

Simplified:

Total latency
≈
prefill time
+
decode time

For long outputs, decode can dominate.

For short outputs, prefill can be more significant.

Therefore optimization depends on workload.

---

## 9. Throughput for LLMs

Can be measured as:

- requests/sec
- tokens/sec

Always clarify:

What is being measured?

And:

Across how many requests/users?

---

## 10. Latency percentiles

Production systems often track:

- P50
- P90
- P95
- P99

Example:

P50 = 100 ms
P95 = 250 ms
P99 = 800 ms

Percentiles reveal tail latency that averages can hide.

---

## 11. Optimization target

"Faster" is not a sufficient optimization goal.

Possible targets:

- lower latency
- higher throughput
- higher utilization
- lower memory usage
- lower cost
- lower power

The target depends on the application.

---

## 12. Systems optimization loop

Profile
→ identify bottleneck
→ define target metric
→ form hypothesis
→ optimize
→ benchmark
→ validate

---

## Key mental model

Latency:
"How long does one request take?"

Throughput:
"How much work can the system complete per unit time?"

Utilization:
"How effectively are hardware resources being used?"

These metrics are related but are NOT the same.

---

## Key takeaway

A system can have:

- low latency but low throughput
- high throughput but high latency
- high hardware utilization but poor latency

Therefore, optimization must always be tied to a specific workload and target metric.
