import matplotlib.pyplot as plt
import numpy as np

# Number of elements
elements = np.array([500_000, 5_000_000, 50_000_000, 500_000_000])

# GPU execution times in milliseconds
gpu_256 = [0.182816, 0.179904, 1.107264, 10.526432]
gpu_512 = [0.277184, 0.180864, 1.109376, 10.524160]
gpu_1024 = [0.200736, 0.183008, 1.121632, 10.506464]

# CPU execution times in milliseconds
cpu = [2.724192, 16.774303, 127.543518, 1243.420044]


# ==========================================
# Graph 1: GPU performance by block size
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(elements, gpu_256, marker='o', linewidth=2,
         label='GPU - Block size 256')

plt.plot(elements, gpu_512, marker='s', linewidth=2,
         label='GPU - Block size 512')

plt.plot(elements, gpu_1024, marker='^', linewidth=2,
         label='GPU - Block size 1024')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Number of Elements')
plt.ylabel('Execution Time (ms)')
plt.title('GPU Vector Addition: Impact of Block Size')

plt.xticks(elements, ['500K', '5M', '50M', '500M'])
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig('gpu_block_size.png', dpi=300)
plt.show()


# ==========================================
# Graph 2: CPU vs GPU performance
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(elements, cpu, marker='o', linewidth=2,
         color='red', label='CPU')

plt.plot(elements, gpu_256, marker='s', linewidth=2,
         label='GPU - Block size 256')

plt.plot(elements, gpu_512, marker='^', linewidth=2,
         label='GPU - Block size 512')

plt.plot(elements, gpu_1024, marker='D', linewidth=2,
         label='GPU - Block size 1024')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Number of Elements')
plt.ylabel('Execution Time (ms)')
plt.title('CPU vs GPU: Vector Addition Performance')

plt.xticks(elements, ['500K', '5M', '50M', '500M'])
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig('cpu_vs_gpu.png', dpi=300)
plt.show()


# ==========================================
# Print GPU speedup over CPU
# ==========================================

print("\nGPU speedup compared with CPU (block size 256):")

for n, cpu_time, gpu_time in zip(elements, cpu, gpu_256):
    speedup = cpu_time / gpu_time
    print(f"{n:>12,} elements: {speedup:.2f}x")