
# GPUs vs CPUs: Architecture and Performance

This lab explores the architectural differences between CPUs and GPUs by implementing vector addition on both processors, measuring performance for different input sizes and GPU block sizes, and fixing a race condition in a CUDA kernel.

## 1. Hardware and System Information

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX PRO 4000 Blackwell |
| Streaming Multiprocessors (SMs) | 70 |
| CUDA cores / Streaming Processors (SPs) | 8,960 |
| CPU | Intel® Xeon® w3-2525 × 16 |
| CPU threads reported by the system | 16 |

GPU specifications: [NVIDIA RTX PRO 4000 Blackwell — TechPowerUp](https://www.techpowerup.com/gpu-specs/rtx-pro-4000-blackwell-workstation.c4279)

## 2. Main Applications of GPUs

### Graphics
GPUs process vertices, pixels, textures, and shading operations in parallel. This makes them essential for real-time rendering, 3D graphics, visualization, and image processing.

### Artificial Intelligence
Training and inference workloads frequently involve large matrix multiplications and tensor operations. These computations can be distributed across many GPU execution units, making GPUs particularly useful for deep learning.

### Other Applications
GPUs are also used for scientific simulations, numerical computing, signal processing, and data-parallel workloads.

GPUs are not automatically faster for every task. Performance depends on the available parallelism, memory traffic, launch overhead, and workload characteristics.

## 3. CPU vs GPU Architecture

A CPU typically has a smaller number of powerful, general-purpose cores. It is designed to handle varied instructions, branching, and latency-sensitive work efficiently.

A GPU is designed for high throughput and can execute many threads performing similar operations concurrently.

CUDA organizes GPU work using the following hierarchy:

- **Thread:** Executes a kernel for a particular element or work item.
- **Block:** Groups threads that can cooperate and share certain resources.
- **Grid:** Contains all blocks launched for a kernel.
- **Streaming Multiprocessor (SM):** Hardware that schedules and executes groups of GPU threads.

In this lab, the GPU has 70 SMs and 8,960 reported CUDA cores, while the CPU has 16 reported threads. These numbers represent different architectural concepts and should not be interpreted as equivalent units.

## 4. Vector Addition Implementation

The experiment computes the element-wise sum of two vectors:

\[
C_i = A_i + B_i
\]

### CPU Implementation

The CPU processes the elements using a loop:

```cpp
for (int i = 0; i < numElements; i++) {
    h_C[i] = h_A[i] + h_B[i];
}
```

### GPU Implementation

The GPU uses a CUDA kernel in which each thread calculates a unique index and adds the corresponding elements:

```cpp
__global__ void vectorAdd(const float *A, const float *B,
                          float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}
```

The grid size is calculated using ceiling division to ensure that enough threads are launched to cover the entire vector. The bounds check handles any additional threads in the final block.

The program verifies the results by comparing each output element against the expected sum.

## 5. Performance Experiments and Speedup

The lab compared CPU and GPU execution times while increasing vector sizes and varying the number of threads per CUDA block.

### Experimental Configurations

| Parameter | Values |
|---|---|
| Vector sizes | 500,000; 5,000,000; 50,000,000; 500,000,000 elements |
| GPU block sizes | 256, 512, 1,024 threads |
| CPU implementation | Element-wise addition using a loop |
| GPU implementation | CUDA vector addition kernel |

The maximum vector size was 500 million elements because this was the largest size that fit the available system RAM during the lab.

CUDA events (`cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, and `cudaEventElapsedTime`) were used to measure GPU execution time. The events surround the kernel launch, so the GPU measurements represent the kernel interval and exclude host-to-device and device-to-host data transfers.

### Performance Results

The plot produced during the lab shows CPU execution time increasing with vector size, while GPU execution times are substantially lower for the larger vectors. The GPU curves for block sizes 256, 512, and 1,024 are close together.

The following values are approximate visual estimates from the plot, not the original raw benchmark output.

| Elements | CPU time (approx.) | GPU time, 1,024 threads/block (approx.) | Approx. speedup |
|---:|---:|---:|---:|
| 500,000 | 2.7 ms | 0.20 ms | 13.5× |
| 5,000,000 | 17 ms | 0.18 ms | 94× |
| 50,000,000 | 130 ms | 1.1 ms | 118× |
| 500,000,000 | 1,250 ms | 10.5 ms | 119× |

### Speedup Calculation

Speedup is calculated as:

\[
\text{Speedup} = \frac{T_{\text{CPU}}}{T_{\text{GPU}}}
\]

A speedup greater than 1 means the GPU measurement is faster than the CPU measurement for the selected timing scope.

### Observations

- The GPU processes large numbers of independent additions in parallel.
- The performance advantage becomes especially visible for larger vectors.
- The tested GPU block-size curves are close together, indicating similar measured execution times for these configurations.
- Vector addition performs little arithmetic per element and moves data through memory, making memory bandwidth important.
- Kernel-only timings exclude data transfers, whereas end-to-end timings include transfer overhead and other relevant operations.

These measurements illustrate the performance trends observed in this lab. The numerical estimates should not be treated as exact benchmark results.

## 6. Race Condition and Its Solution

The second program counts the number of elements in a vector whose values exceed a threshold of `0.5`.

A CPU loop computes the reference count, while a CUDA kernel performs the same operation in parallel.

### Understanding the Race Condition

Multiple GPU threads may find qualifying values simultaneously and attempt to increment the same counter.

A normal increment such as:

```cpp
(*count)++;
```

is a read-modify-write operation. If several threads execute these operations concurrently, updates can overwrite one another, causing the final count to be incorrect.

This is a race condition: the result can depend on the timing of concurrent memory accesses.

### Solution Using `atomicAdd`

The corrected CUDA kernel uses:

```cpp
atomicAdd(count, 1);
```

The atomic operation ensures that each counter increment is performed safely, preventing simultaneous updates from losing increments.

### Validation

The program:

1. Creates a vector containing 50,000 random floating-point values.
2. Uses `0.5` as the threshold.
3. Computes the expected count on the CPU.
4. Copies the input vector to GPU memory.
5. Resets the GPU counter and runs the kernel five times.
6. Compares each GPU result against the CPU reference.

The test passes when all five GPU counts match the CPU reference.

Atomic operations are a common way to protect shared counters in parallel programs. Other approaches include per-block partial counts followed by a reduction, which can reduce contention when many threads update the same counter.

