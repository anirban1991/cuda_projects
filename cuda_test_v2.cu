// Device code
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
 int i = blockDim.x * blockIdx.x + threadIdx.x;
 if (i < N)
 C[i] = A[i] + B[i];
}
// Host code
int main()
{
 int N = 5;
 size_t size = N * sizeof(float);
 // Allocate input vectors h_A and h_B in host memory
//  float* h_A = (float*)malloc(size);
//  float* h_B = (float*)malloc(size);
 // Initialize input vectors
float h_A[] = {1.5,2.0,4.1,5.2,6.0};
float h_B[] = {1.5, 1.5, 3.0, 3.0, 4.0};
float h_C[5];
 // Allocate vectors in device memory
 float* d_A;
 cudaMalloc(&d_A, size);
 float* d_B;
 cudaMalloc(&d_B, size);
 float* d_C;
 cudaMalloc(&d_C, size);
 // Copy vectors from host memory to device memory
 cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
 cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
 // Invoke kernel

 int threadsPerBlock = 256;
 int blocksPerGrid =
 (N + threadsPerBlock - 1) / threadsPerBlock;
 VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
 // Copy result from device memory to host memory
 // h_C contains the result in host memory
 cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
 // Free device memory
 cudaFree(d_A);
 cudaFree(d_B);
 cudaFree(d_C);
 // Free host memory
}
