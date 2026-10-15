#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main() {
    int numElements = 5000000;

    float a[numElements], b[numElements], c[numElements];

    // Initialize input arrays
    for (int i = 0; i < numElements; i++) {
      a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }

    // Measure CPU execution time
    clock_t start = clock();

    // Vector addition on CPU
    for (int i = 0; i < numElements; i++) {
        c[i] = a[i] + b[i];
    }

    clock_t end = clock();

    // Calculate execution time in milliseconds
    double cpu_time_ms =
        ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    // Print results
    printf("CPU Vector Addition of 50000 elements\n");
    printf("First result: %f\n", c[0]);
    printf("CPU execution time: %.6f ms\n", cpu_time_ms);

    return 0;
}