# LLM Inference Optimization Lab

## Overview

A reproducible single-GPU LLM inference benchmarking framework for studying latency, throughput, memory usage, and scaling behavior.

V0.2 focuses on separating first-token latency from steady-state decode performance using:

- E2E latency
- TTFT
- TPOT
- E2E throughput
- Decode throughput
- Peak VRAM

Current benchmark setup:

- Backend: Hugging Face Transformers
- Model: `Qwen/Qwen3.5-4B`
- Precision: BF16
- GPU: NVIDIA GeForce RTX 4090
- Batch size: 1

---

## Reproduction

### Environment

- Python 3.11
- PyTorch 2.14.0 + CUDA 13.0
- Transformers 5.17.0
- NVIDIA GeForce RTX 4090

Install dependencies:

```bash
uv sync
```

Run benchmark:

```bash
cd src
uv run python -m benchmark
```

Generate analysis figures:

```bash
uv run python analysis/plot_context_scaling.py \
  --input results/raw/context_scaling_<timestamp>.json

uv run python analysis/plot_output_length_scaling.py \
  --input results/raw/output_length_scaling_<timestamp>.json
```

---
## RoadMaps
```text
V0.x → V1.0  Benchmarking / Architecting
│
├── V0.1  HF baseline ✅
│         E2E / throughput / VRAM
│         context scaling
│
├── V0.2  Fine-grained latency ✅
│         TTFT / TPOT
│         prefill vs decode
│         output-length scaling
│
├── V0.3  Batch scaling ✅
│         batch size vs latency / throughput / VRAM
│
├── V0.4  Backend comparison  ✅
│         HF vs vLLM vs (Todo: SGLang🚧)
│         offline + serving workload
│
├── V0.5  Precision / quantization 🚧
│         FP16 / BF16 / INT8 / 4-bit
│         memory / latency / throughput
│         + quality drift
│
├── V0.6  Model scaling
│         model size
│         Qwen vs Mistral
│
├── V0.7  VLM / multimodal inference
│         image tokens
│         multimodal TTFT / memory / throughput
│
├── V0.8  Serving workload
│         concurrency
│         ISL / OSL distributions
│         queueing / request scheduling
│
├── V0.9  Capacity / reporting
│         benchmark reports
│         capacity estimation
│         Docker / reproducibility
│
└── V1.0  Reproducible inference benchmark suite  
          multiple backends
          multiple workloads
          reproducible reports


V1.x  Profiling & Optimization 📖📝🤔💭
│
├── PyTorch Profiler / Nsight
├── prefill / decode kernel attribution
├── attention / KV-cache analysis
├── torch.compile
├── memory bandwidth / compute bottlenecks
├── MoE routing / expert utilization
├── Triton operators
├── CUDA operators
├── kernel fusion
└── quantized kernels
```
---

## Metrics

Let:

- `t0` = request start
- `t1` = first generated token observed
- `t2` = generation complete
- `N` = number of generated tokens

### E2E latency

```text
E2E = t2 - t0
```

### TTFT

```text
TTFT = t1 - t0
```

### Decode time

```text
Decode Time = E2E - TTFT
```

### TPOT

```text
TPOT = (E2E - TTFT) / (N - 1)
```

### Decode throughput

```text
Decode TPS = (N - 1) / (E2E - TTFT)
```

### E2E throughput

```text
E2E TPS = N / E2E
```

Therefore:

```text
Decode TPS ≈ 1000 / TPOT_ms
```

The request latency can be approximated as:

```text
T_request ≈ TTFT + (N - 1) × TPOT
```

---

## Context Scaling

Fixed configuration:

```text
output_tokens = 128
batch_size = 1
dtype = BF16
```

Varied input length:

```text
input_tokens = 128, 512, 1024, 2048
```

### Result

```text
Input length ↑
→ TTFT ↑ strongly
→ E2E latency ↑
→ E2E throughput ↓
→ Peak VRAM ↑
→ TPOT ≈ stable
→ Decode throughput ≈ stable
```

Measured TTFT:

```text
128 tokens  →  28.3 ms
512 tokens  →  60.1 ms
1024 tokens → 111.5 ms
2048 tokens → 224.0 ms
```

TPOT remains around:

```text
~15.2 ms/token
```

Increasing context length primarily increases first-token / prefill cost, while steady-state decode performance remains comparatively stable.

---

## Output Length Scaling

Fixed configuration:

```text
input_tokens = 512
batch_size = 1
dtype = BF16
```

Varied output length:

```text
output_tokens = 16, 32, 64, 128, 256
```

### Result

```text
Output length ↑
→ E2E latency ↑ approximately linearly
→ TTFT ≈ constant
→ TPOT ≈ stable
→ Decode throughput ≈ stable
→ E2E throughput approaches decode throughput
```

Measured TTFT stays around:

```text
~60 ms
```

TPOT remains around:

```text
~15.1 ms/token
```

As output length grows:

```text
E2E TPS → Decode TPS
```

because the fixed TTFT overhead is amortized over more generated tokens.

---
## Backend Benchmarking

### HF vs vLLM

Test configuration:

- Model: Qwen/Qwen3.5-4B
- GPU: RTX 4090
- Precision: BF16
- Batch size: 1
- Input tokens: 512
- Output tokens: 128

| Backend | Mean E2E Latency | Mean E2E Throughput |
|---|---:|---:|
| Hugging Face | 1988.0 ms | 64.38 tok/s |
| vLLM | 1293.9 ms | 98.93 tok/s |

Under this specific workload, vLLM reduced E2E latency by ~34.9% and increased E2E throughput by ~53.7% compared with the Hugging Face baseline.

---

## V0.2 Test Summary

The benchmark separates LLM inference latency into two dominant components:

```text
Input length
→ primarily affects TTFT / prefill cost

Output length
→ primarily affects total decode duration
```

For the tested configuration:

```text
T_request ≈ TTFT(L_input) + (N_output - 1) × TPOT
```

with TPOT remaining relatively stable across both experiments.

## v0.4 — Multi-Backend Benchmarking

The benchmark framework now supports backend-specific execution behind a shared experiment interface.

Currently supported:

- Hugging Face Transformers
- vLLM
- SGLang — planned

The runner reuses long-lived backend resources within an experiment group and explicitly releases them when switching backends.

Backend comparison experiments keep the following inputs fixed:

- Model
- Precision
- Tokenized input workload
- Batch size
- Input sequence length
- Output sequence length
- Decoding constraints

Prefix caching is disabled for controlled backend comparisons to avoid reusing computation across repeated requests.
