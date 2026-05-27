# Axiom-Parallel: Multi-Dimensional LLM Distributed Training Infrastructure

A highly structured, production-ready framework implementing multi-dimensional (3D) parallelism strategies for Large Language Models. This repository contains the core communication primitives, custom layer abstractions, and execution pipelines required to scale models that exceed individual GPU memory thresholds.

---

## 📌 Table of Contents

- [The Problem: The Memory & Compute Wall](#the-problem-the-memory--compute-wall)
- [The Solution: Unified 3D Parallelism](#the-solution-unified-3d-parallelism)
- [Architecture & Directory Structure](#architecture--directory-structure)
- [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
- [Performance Benchmarking & Instrumentation](#performance-benchmarking--instrumentation)

---

## 💡 The Problem: The Memory & Compute Wall

As Large Language Models scale past tens of billions of parameters, training or serving them on a single hardware accelerator becomes mathematically impossible due to two structural constraints:

- **The Memory Wall:** An `80GB` GPU cannot simultaneously hold model weights, optimizer states (e.g., Adam's FP32 copies), gradients, and activation tensors.
- **The Compute Wall:** The trillions of floating-point operations (`FLOPs`) required for convergence make sequential or single-device execution logistically non-viable.

### Traditional Bottlenecks

| Bottleneck | Root Cause |
|---|---|
| **Naive Data Parallelism** | The entire model must fit on every single GPU |
| **Unoptimized Tensor Splitting** | Constant synchronization causes massive network latency during forward and backward passes |
| **Pipeline Bubbles** | Poor execution scheduling leads to idle GPUs and severe resource underutilization |

---

## 🛠️ The Solution: Unified 3D Parallelism

This repository implements **Multi-Dimensional (3D) Parallelism**, decoupling the model across three independent but complementary structural axes to maximize hardware utilization (MFU) and communication efficiency.

### 1. Tensor Parallelism (TP)

**Strategy:** Intra-layer splitting across the hidden dimension, based on Megatron-style architectures.

**Mechanism:**
- Splits the **Column Parallel Linear** layer (`W_gate`, `W_query`) to avoid communication before activation functions.
- Splits the **Row Parallel Linear** layer (`W_down`, `W_out`) and aggregates results via a single, efficient **All-Reduce** operation at the layer boundary.

### 2. Pipeline Parallelism (PP)

**Strategy:** Inter-layer splitting that partitions model layers sequentially across distinct devices.

**Mechanism:**
- Implements a **1F1B (One Forward, One Backward)** schedule where micro-batches are pipelined concurrently.
- Devices exchange activation tensors and gradients only at boundaries via **point-to-point (P2P)** communication, significantly reducing the activation memory footprint.

### 3. Data Parallelism (DP) & ZeRO

**Strategy:** Inter-node scaling by replicating parallelized model chunks across data shards.

**Mechanism:**
- Integrates **Zero Redundancy Optimizer (ZeRO)** concepts to shard optimizer states and gradients across data-parallel ranks, eliminating duplicate memory allocations.

---

## 📂 Architecture & Directory Structure

```
├── configs/               # Topology definitions (e.g., TP=2, PP=2, DP=2)
├── src/
│   ├── communication/     # Custom collective primitives (All-Reduce wrappers, P2P ring)
│   ├── layers/            # ColumnParallelLinear and RowParallelLinear PyTorch modules
│   ├── schedules/         # 1F1B and interleaved pipeline execution logic
│   └── models/            # Core transformer blocks optimized for distributed splitting
├── main.py                # Distributed runtime initialization and benchmarking engine
└── requirements.txt       # Core dependencies (PyTorch, NCCL, etc.)
```

---

## 🚀 Step-by-Step Implementation Guide

### 1. Environment Verification

Ensure your environment recognizes your cluster topology and has the NCCL backend properly configured:

```bash
python -c "import torch; print(f'GPUs Available: {torch.cuda.device_count()}, NCCL Available: {torch.distributed.is_nccl_available()}')"
```

### 2. Installation

Clone the repository and set up a localized Python environment:

```bash
git clone https://github.com/rohantikotekar/llm-parallelism.git
cd llm-parallelism
pip install -r requirements.txt
```

### 3. Setting the Topology Configuration

Define your cluster topology matrix inside a configuration file (e.g., `configs/3d_mesh_8gpus.yaml`). For an 8-GPU cluster, a typical 3D partition looks like this:

```yaml
# Cluster Topology Matrix
parallelism:
  tensor_parallel_size: 2
  pipeline_parallel_size: 2
  data_parallel_size: 2  # Total Ranks = TP * PP * DP = 8

model:
  hidden_size: 4096
  num_layers: 32
  num_heads: 32
```

### 4. Launching the Multi-GPU Cluster

Execute the distributed orchestrator using PyTorch's native elastic launch utility (`torchrun`).

**Single Node, 8-GPU Run:**

```bash
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    main.py --config configs/3d_mesh_8gpus.yaml
```

**Multi-Node Run (Example for Node 0):**

```bash
torchrun \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="10.0.0.1" \
    --master_port=29500 \
    --nproc_per_node=8 \
    main.py --config configs/3d_mesh_8gpus.yaml
```

---

## 📊 Performance Benchmarking & Instrumentation

The framework logs detailed telemetry to balance processing speed against communication bottlenecks:

| Metric | Description |
|---|---|
| **TFLOPs/sec per GPU** | Tracks raw calculation efficiency against the theoretical hardware maximum |
| **Comm/Comp Ratio** | Quantifies time spent on cross-device network transfers (`NCCL_AllReduce`, `P2P_Send_Recv`) versus actual matrix calculations |
| **Memory Tracking** | Provides precise per-GPU peak VRAM breakdown across weights, gradients, and activation buffers |

---

## 📄 License

This project is open-source. See [`LICENSE`](LICENSE) for details.
