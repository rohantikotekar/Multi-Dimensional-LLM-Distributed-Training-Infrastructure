# Axiom-Parallel: Shared-Memory LLM Parallelism with OpenMP

A project exploring how far you can push GPT-2 (124M parameters) training on a multi-core CPU using OpenMP — parallelizing the backward pass, fixing cache behavior, and measuring the real speedup with `perf`.

---

## 📌 Table of Contents

- [The Problem: The Backpropagation Bottleneck](#the-problem-the-backpropagation-bottleneck)
- [The Solution: Targeted OpenMP Optimizations](#the-solution-targeted-openmp-optimizations)
- [Performance Baseline & Target Results](#performance-baseline--target-results)
- [Architecture & Directory Structure](#architecture--directory-structure)
- [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)

---

## 💡 The Problem: The Backpropagation Bottleneck

Training even a moderately sized LLM on a CPU is painfully slow — not because the hardware is weak, but because the default sequential implementation doesn't use most of it.

Profiling the baseline with `linux perf record` makes this obvious pretty quickly: `matmul_backward` accounts for the overwhelming majority of runtime. Everything else is noise by comparison.

### Why it's slow

| Bottleneck | What's happening |
|---|---|
| **Write-sharing between threads** | When multiple threads update the same gradient array positions, they constantly block each other waiting for locks |
| **Poor cache behavior** | The large matrix dimensions in GPT-2's hidden layers mean memory accesses jump around a lot, trashing L1/L3 cache lines |
| **Cores sitting idle** | Without explicit parallelism, a 12-core machine is doing the work of one — the other 11 are just waiting |

---

## 🛠️ The Solution: Targeted OpenMP Optimizations

All changes live in `my_train_gpt2.h`. The goal was to speed up `matmul_backward` specifically, since that's where the time actually goes.

### 1. Multi-Core Loop Scheduling

The outer loops in the matrix multiply are parallelized with `#pragma omp parallel for`, splitting the work evenly across all available threads. For GPT-2's hidden dimensions this gives a near-linear reduction in wall time as thread count goes up.

### 2. Synchronization Management

The tricky part of parallelizing backward passes is gradient accumulation — multiple threads writing to the same output positions. This is handled by restructuring loops to minimize sharing, using private accumulators where possible, and falling back to `#pragma omp critical` only where necessary.

### 3. Loop Tiling & SIMD Vectorization

- **Tiling:** Reorders the loop structure so each thread works through a contiguous block of the matrix, keeping data hot in cache rather than constantly evicting and reloading it.
- **SIMD:** `#pragma omp simd` lets the compiler map inner loop iterations to vector instructions, so each core processes multiple floats per clock instead of one.

---

## 📊 Performance Baseline & Target Results

| Metric | Baseline (`test_gpt2` / `train_gpt2`) | Optimized (`my_test_gpt2` / `my_train_gpt2`) |
|---|---|---|
| **Speedup Target** | 1.0× (reference) | ≥ 3.5× (cluster target) |
| **Cache Miss Rate — `matmul_backward`** | ~16.0% | Reduced via loop tiling |
| **Cache Miss Rate — System-wide** | ~8.2% | Reduced via access reordering |
| **Threads Used** | 1 (sequential) | 12 (default) |
| **Total Training Time** | ~6–10 minutes | ~3× faster than baseline |

---

## 📂 Architecture & Directory Structure

```
├── src/
│   ├── layers/            # Transformer layer implementations
│   ├── communication/     # Shared-memory buffer management
│   └── architecture/      # Multi-core scaling configurations
├── train_gpt2.c           # Original sequential training loop
├── train_gpt2.h           # Original sequential primitives (untouched)
├── my_train_gpt2.h        # Parallelized implementations go here
├── test_gpt2.c            # Tests for the sequential baseline
├── my_test_gpt2.c         # Tests for the parallel version
└── Makefile               # Build rules — pass OPENMP=yes to enable threading
```

---

## 🚀 Step-by-Step Implementation Guide

### 1. Environment Setup

Make sure OpenMP is enabled before building anything:

```bash
export OPENMP=yes
```

### 2. Installation

```bash
git clone https://github.com/rohantikotekar/llm-parallelism.git
cd llm-parallelism
```

### 3. Build and Run the Tests

Start here before running full training — the test binaries are faster to build and will catch any correctness issues in your parallel implementation:

```bash
# Build both versions
make clean test_gpt2 my_test_gpt2 OPENMP=yes

# Run the sequential baseline
./test_gpt2

# Run your parallel version and compare outputs
./my_test_gpt2
```

If the outputs match, you're good to move on.

### 4. Run Full Training

Once the tests pass, run the complete training loop to see the real-world speedup:

```bash
make clean my_train_gpt2 OPENMP=yes
./my_train_gpt2
```

Watch the per-step timing — with 12 threads and the optimizations in place, you should see roughly 3× faster iteration times compared to the sequential baseline.

---

## 📄 License

Open-source. See [`LICENSE`](LICENSE) for details.
