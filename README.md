# Axiom-Parallel: Shared-Memory LLM Parallelism with OpenMP

A modular framework designed to explore, profile, and optimize shared-memory execution topologies for Large Language Models (LLMs) using pure C and OpenMP. By implementing compiler directives, loop transformations, and memory optimizations, this project addresses hardware bottlenecks during the backpropagation pass of a reproduced GPT-2 (124M parameter) model.

---

## 📌 Table of Contents

- [The Problem: The Backpropagation Bottleneck](#the-problem-the-backpropagation-bottleneck)
- [The Solution: Targeted OpenMP Optimizations](#the-solution-targeted-openmp-optimizations)
- [Architecture & Directory Structure](#architecture--directory-structure)
- [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
- [Performance Tracking & System Telemetry](#performance-tracking--system-telemetry)

---

## 💡 The Problem: The Backpropagation Bottleneck

When training Large Language Models on multi-core CPU architectures, matrix arithmetic dominates execution time. Naive sequential computation engines cannot keep up with the vast number of floating-point operations (`FLOPs`) required for network parameter updates.

Through systemic execution profiling via `linux perf record`, the computational profile shows that `matmul_backward` is the most time-consuming primitive.

### Key Performance Barriers

| Bottleneck | Root Cause |
|---|---|
| **Data Write-Sharing Overhead** | Multiple processing units updating identical tensor element positions create severe multi-thread synchronization blocks |
| **Cache Locality Degradation** | Large hidden dimensions introduce non-contiguous memory access sequences, leading to costly L1/L3 CPU cache miss rates |
| **Idle Resource Cores** | Unparallelized loops leave powerful multi-socket CPU systems underutilized, lowering execution efficiency |

---

## 🛠️ The Solution: Targeted OpenMP Optimizations

This codebase speeds up the backward pass by applying standard compiler engineering and hardware-aware optimizations to `my_train_gpt2.h`.

### 1. Multi-Core Loop Scheduling

Using `#pragma omp parallel for`, outer execution loops are distributed across independent system threads. This divides massive hidden dimensions into balanced iterations, minimizing scheduling sync penalties.

### 2. Synchronization Management

Memory locations are safely managed using localized loop structures, private variables, or strict critical blocks (`#pragma omp critical`) — ensuring data-sharing updates to target arrays proceed without throughput degradation.

### 3. Loop Tiling & SIMD Vectorization

- **Temporal Locality:** Loops are restructured to reuse data blocks inside CPU cache lines, directly reducing data miss rates.
- **Vector Units:** `#pragma omp simd` enables single-instruction, multiple-data (SIMD) instruction mapping, allowing individual processors to process multiple floating-point values concurrently.

---

## 📂 Architecture & Directory Structure

```
├── src/
│   ├── layers/            # Core transformer layers optimized for distributed arrays
│   ├── communication/     # Shared-memory message passing and buffer wrappers
│   └── architecture/      # Structural configurations for multi-core scaling
├── train_gpt2.c           # Main baseline training loop orchestrator
├── train_gpt2.h           # Original sequential mathematical primitives
├── my_train_gpt2.h        # Optimized multi-threaded execution implementations
├── test_gpt2.c            # Validation engine for sequential computation checks
├── my_test_gpt2.c         # Validation engine for optimized parallel tracking
└── Makefile               # Build script with OpenMP compilation parameters
```

---

## 🚀 Step-by-Step Implementation Guide

### 1. Environment & Dependencies Setup

Verify your environment includes the standard compiler flags necessary to build multi-threaded code blocks:

```bash
export OPENMP=yes
```

### 2. Installation

Clone and set up the localized project repository:

```bash
git clone https://github.com/rohantikotekar/llm-parallelism.git
cd llm-parallelism
```

### 3. Compiling the Benchmarks

To test optimizations without executing full training sequences, build the scaled-down profiling targets:

```bash
# Build both baseline and custom parallel binaries
make clean test_gpt2 my_test_gpt2 OPENMP=yes
```

### 4. Running Execution Validations

Execute the test files to verify your parallelized implementation against the sequential baseline:

```bash
# Evaluate baseline output profiles
./test_gpt2

# Evaluate multi-threaded execution speeds
./my_test_gpt2
```

### 5. Running the Complete Training Engine

Once local validations pass, launch the full training run to benchmark model speedups across multiple processing threads:

```bash
make clean my_train_gpt2 OPENMP=yes
./my_train_gpt2
```

---

## 📊 Performance Tracking & System Telemetry

The runtime architecture records detailed profiling metrics to measure and analyze processing speedups:

| Metric | Description |
|---|---|
| **Core Execution Time (ET)** | Measures real-world wall time to evaluate processing speed gains |
| **Cache Telemetry Matrix** | Monitors data loading trends to quantify the reduction in cache miss rates |
| **Thread Scaling Efficiency** | Profiles how well execution speeds up across different thread counts (e.g., targeting 12-thread scaling) |

---

## 📄 License

This project is open-source. See [`LICENSE`](LICENSE) for details.
