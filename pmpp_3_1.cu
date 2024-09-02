#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__
void arrayAddKernel(float* A, float* B, float *C, int n, size_t pitch)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy =  blockIdx.y * blockDim.y + threadIdx.y;


    if ((tidx<n) && (tidy <n))
    {
        float *row_a = (float *)((char*)A + tidy * pitch);
        float *row_b = (float *)((char*)B + tidy * pitch);
        float *row_c = (float *)((char*)C + tidy * pitch);
        row_c[tidx] =  row_a[tidx] + row_b[tidx];
        // printf("A value %.3f, %d, %d \n", row_a[tidx], tidx, tidy);
        // printf("B value %.3f, %d, %d \n", row_b[tidx], tidx, tidy);
        // printf("C value %.3f, %d, %d \n", row_c[tidx], tidx, tidy);

    }


}

// void arrayAddKernel(float* A, float* B, float *C, int n, size_t pitch)
// {
//     int tidx = threadIdx.x * blockDim.x + blockIdx.x;
//     int tidy = threadIdx.y * blockDim.y + blockIdx.y;


//     if ((tidx<n) && (tidy <n))
//     {
//         float *row_a = (float *)((char*)A + tidy * pitch);
//         float *row_b = (float *)((char*)B + tidy * pitch);
//         float *row_c = (float *)((char*)C + tidy * pitch);
//         row_c[tidx] =  row_a[tidx] + row_b[tidx];
//         printf("A value %.3f, %d, %d \n", row_a[tidx], tidx, tidy);
//         printf("B value %.3f, %d, %d \n", row_b[tidx], tidx, tidy);
//         printf("C value %.3f, %d, %d \n", row_c[tidx], tidx, tidy);

//     }


// }

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void array_add(float* A, float* B, float* C, int n)
{
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;
    size_t pitch;
    // cudaError_t error;

    cudaMallocPitch((void **)&d_A, &pitch, size,n);
    cudaMemcpy2D(d_A, pitch, A, size, size, n, cudaMemcpyHostToDevice);
    cudaMallocPitch((void **)&d_B, &pitch, size,n);
    cudaMemcpy2D(d_B, pitch, B, size, size, n, cudaMemcpyHostToDevice);
    cudaMallocPitch((void **)&d_C, &pitch, size,n);
    printf("hello0\n");
    // Kernel Code
    dim3 grid(16,16);
    dim3 block(16,1);

    arrayAddKernel<<< grid, block >>> (d_A, d_B, d_C, n, pitch );
    // size_t pitch = sizeof(float) * n;
    printf("size in bytes %d",size);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy2D(C, size, d_C, pitch, size, n, cudaMemcpyDeviceToHost);

    // for (int i= 0; i<size ;i++)
    // { for (int j = 0 ; j< size; j++)
    // {
    //     printf("%.3f ", C);
    // }
    // }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


}

int main()
{
    int dim = 3;
    float h_A[3][3] = {{2.5,3.5,5.0},{1.5,3.5,1.0}, {0.0,0.0,0.0}};
    float h_B[3][3] = {{1.5,3.5,0.0},{2.5,3.0,1.5}, {0.0,0.0,0.0}};
    float h_C[3][3];



    array_add((float *)h_A, (float *)h_B, (float *)h_C, dim);

    // NOT getting any output here 

    for (int i= 0; i<dim ;i++)
    { for (int j = 0 ; j< dim; j++)
    {
        printf("%.1f ", h_C[i][j]);
    }
    }

}