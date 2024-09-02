#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__
// void arrayAddKernel(float* A, float* B, float *C, int n)
// {
//     int tidx = threadIdx.x;
//     int tidy =  blockIdx.x;


//     if ((tidx<n) && (tidy <n))
//     {
//         C[n*tidy+tidx] = A[n*tidy+tidx] + B[n*tidy+tidx];

//     }


// }

// void arrayAddKernel(float* A, float* B, float *C, int n)
// {
//     // int i = blockDim.x * blockIdx.x + threadIdx.x;
//     // int i = blockIdx.x;
//     // int i = blockDim.x + threadIdx.x;
//     int i = blockDim.x * blockIdx.x + threadIdx.x + 3;


//     if (i< n*n)
//     {
//         C[i] = A[i] + B[i];
//         printf("%.1f", C[i]);

//     }


// }

void arrayAddKernel(float* A, float* B, float *C, int n)
{

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;


    if ((row < n) && (col < n))
    {
        C[row*n + col] = A[row*n+col] + B[row*n+col];
        printf("%.1f", C[row*n + col]);

    }


}


void array_add(float* A, float* B, float* C, int n)
{
    int size = n*n*sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaError_t error;
    // size_t pitch;

    cudaMalloc((void **)&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_B, size);
    cudaMemcpy(d_B, B, size,cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_C,  size);
    printf("hello0\n");
    // Kernel Code
    dim3 grid(16,16);
    dim3 block(16,16);

    

    arrayAddKernel<<< grid, block >>> (d_A, d_B, d_C, n);

    error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
        exit (EXIT_FAILURE);
    }

    // size_t pitch = sizeof(float) * n;
    // printf("size in bytes %d",size);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

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

    int dev_count;
    cudaGetDeviceCount(&dev_count);

    cudaDeviceProp dev_prop;
for (int i=0; i < dev_count; i++) {
cudaGetDeviceProperties( &dev_prop, i);
printf("%s\n",dev_prop);
printf("%d\n",dev_prop.maxThreadsPerBlock);
printf("%d\n",dev_prop.multiProcessorCount);
printf("%d\n",dev_prop.clockRate);
printf("%d\n",dev_prop.maxGridSize);
// decide if device has sufficient resources and capabilities
}

    printf("%d\n",dev_count);




    array_add((float *)h_A, (float *)h_B, (float *)h_C, dim);

    // NOT getting any output here 

    for (int i= 0; i<dim ;i++)
    { for (int j = 0 ; j< dim; j++)
    {
        printf("row %d column %d -  %.1f ", i, j, h_C[i][j]);
    }
    }

}