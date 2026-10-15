#include <stdio.h>
#include <stdlib.h>

// CUDA kernel function to add two vectors
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    // Get the unique thread ID, which is the index in the vector
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char* argv[]) {
    bool cpu = (argc > 1) && (argv[1][0] == 'c');

    // Vector size and memory size
    long int numElements[4] = {500000, 5000000, 50000000, 500000000};
    float time[4] = {0, 0,  0, 0};
    for(int s=0; s<4; s++)
    {

    size_t size = numElements[s] * sizeof(float);
    
    printf("Vector addition of %ld elements\n", numElements[s]);
    
    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    /*if(h_A == NULL || h_B == NULL || h_C = NULL) {
        fprintf(stderr, "Host memory allocation failed!\n");
        return EXIT_FAILURE;
    }*/
    
    // Initialize host arrays with random data
    for (int i = 0; i < numElements[s]; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Launch the CUDA kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElements[s] + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if(cpu) {
        for (int i = 0; i < numElements[s]; i++) {
            h_C[i] = h_A[i] + h_B[i];
        }
    }
    else vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements[s]);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU execution time: %f ms\n", milliseconds);
    time[s] = milliseconds;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy result back to host
    if(!cpu) cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify the result
    for (int i = 0; i < numElements[s]; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("Done\n");

    }


    for(int i=0; i<4; i++)
        printf("%f: ", time[i]);
    return 0;
} 
