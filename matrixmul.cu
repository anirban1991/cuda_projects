
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Defining block width as compile time constant
#define BLOCKWIDTH 16


// Defining error check macro for gpu codes
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



__global__ void matmulKernel (float* first_matrix, float* second_matrix, float* output_matrix, int row_width_first, int col_width_first, int row_width_second, int col_width_second)
{
// row index of the first matrix
 int row = blockDim.x * blockIdx.x + threadIdx.x;
// column index of the second matrix 
 int col = blockDim.y * blockIdx.y + threadIdx.y;

if ((row < row_width_first) && (col < col_width_second))
{
     __syncthreads();
    float pValue = 0;
    for (int i = 0; i < col_width_first; i++)
    {
        printf("%d\n",i);
        printf("first element %d , %.2f\n",row*col_width_first+i, first_matrix[row*col_width_first+i]);
        printf("second element %d, %.2f\n",col + row_width_second*i, second_matrix[col + row_width_second*i]);
        
        
        pValue = pValue + (first_matrix[row*col_width_first+i] * second_matrix[col + col_width_second*i]);
        printf("pvalue %.2f\n",pValue);
    }
    output_matrix[row*col_width_second+col] = pValue;
    printf("final elementat %d is %.2f\n",row*col_width_first+col, output_matrix[row*col_width_first+col]);

}
 
 }

 void helperFunction (float* A, float* B, float* C, int first_row, int first_col, int second_row, int second_col){

    int first_array_dim = first_row *first_col;
    int second_array_dim = second_row*second_col;
    int output_array_dim = first_row*second_col;
    int size_first, size_second, size_output, size;
    size_first = first_array_dim * sizeof(float);
    size_second = second_array_dim * sizeof(float);
    size_output = output_array_dim * sizeof(float);
    float *d_A, *d_B, *d_C;
    // size_t pitch;
    if (size_first > size_second)
    {
        size = size_first;
    }
    else
    {
        size = size_second;
    }



    gpuErrchk(cudaMalloc((void **)&d_A, size_first));
    gpuErrchk(cudaMemcpy(d_A, A, size_first, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void **)&d_B, size_second));
    gpuErrchk(cudaMemcpy(d_B, B, size_second,cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void **)&d_C,  size_output));
    // Kernel Code
    dim3 grid(ceil(size/BLOCKWIDTH),1,1);
    dim3 block(BLOCKWIDTH, BLOCKWIDTH, 1);

    matmulKernel<<< grid, block >>> (d_A, d_B, d_C, first_row, first_col, second_row, second_col);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(C, d_C, size_output,cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
 }




int main()
{
    int first_matrix_row = 3;
    int first_matrix_col = 2;
    int second_matrix_row = 2;
    int second_matrix_col = 2;

    float h_A[3][2] = {{1.0,2.0},{3.0,4.0},{5.0,6.0}};
    float h_B[2][2] = {{1.0,2.0},{3.0,4.0}};
    float h_C[3][2];


    // int first_matrix_row = 2;
    // int first_matrix_col = 2;
    // int second_matrix_row = 2;
    // int second_matrix_col = 2;

    // float h_A[3][2] = {{2.0,3.0},{1.0,3.0}};
    // float h_B[2][2] = {{1.0,3.0},{2.0,3.0}};
    // float h_C[2][2];

    // int first_matrix_row = 2;
    // int first_matrix_col = 3;
    // int second_matrix_row = 3;
    // int second_matrix_col = 2;

    // float h_A[2][3] = {{1.0,2.0,3.0},{3.0,4.0,5.0}};
    // float h_B[3][2] = {{1.0,2.0},{3.0,4.0},{5.0,6.0}};
    // float h_C[2][2];


    helperFunction((float *)h_A, (float *)h_B, (float *)h_C, first_matrix_row, first_matrix_col, second_matrix_row, second_matrix_col);


    for (int i= 0; i<first_matrix_row ;i++)
    { for (int j = 0 ; j< second_matrix_col; j++)
    {
        printf("row %d column %d -  %.2f ", i, j, h_C[i][j]);
    }
    }

}