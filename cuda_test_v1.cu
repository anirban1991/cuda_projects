
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


// #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
// #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
// #endif


__global__ void vecAddKernel(float* A, float* B, float* C, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i <n) 
    {
        C[i] = A[i] + B[i];
        // printf("Thread %d value %.3f,", threadIdx.x, C[i]);
        
    }
    
    
}

void vecAdd(float* A, float* B, float* C, int n)
{
    //  int i ;
    // for (i= 0; i<n ;i++)
    // {
    //     h_C[i] = h_A[i] +h_B[i];
    // }

    // for (i= 0; i<n ;i++)
    // {
    //     printf("%.3f,", h_C[i]);
    //     // h_C[i] = h_A[i] +h_B[i];
    // }

    int size = n* sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_C, size);
    vecAddKernel <<< 100, 256>>> (d_A, d_B, d_C, n);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);



    

}



int main ()
{
   float h_A[] = {1.5,2.0,4.1,5.2,6.0};
   float h_B[] = {1.5, 1.5, 3.0, 3.0, 4.0};
   float h_C[5];
   int N = 5;
   printf("calling the wrapper function\n");

   vecAdd(h_A, h_B, h_C, N);
   for (int i= 0; i<N ;i++)
    {
        printf("%.3f,", h_C[i]);
    }

   printf("ending the wrapper function\n");



   
}