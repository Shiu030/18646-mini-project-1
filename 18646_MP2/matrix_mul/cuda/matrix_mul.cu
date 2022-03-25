/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <cuda.h>
#include <cuda_runtime.h>
#include "matrix_mul.h"
#define BLOCK_WIDTH 2

namespace cuda
{
  __global__ void matrix_mul_kernel(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, int sq_dimension)
  {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float temp_result = 0;
    
    for(int k = 0; k < sq_dimension; k++)
    {	
	  temp_result += sq_matrix_1[row * sq_dimension + k] * sq_matrix_2[k * sq_dimension + col];
    }

    sq_matrix_result[row * sq_dimension + col] += temp_result;
  }
  
  void matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension)
  {
    int size = sq_dimension * sq_dimension * sizeof(float);
    float *sq_matrix_1_d, *sq_matrix_2_d, *sq_matrix_result_d;
    
    /***************************************************
    Step 1: Allocation of memory on device memory  
    ****************************************************/
    
    /* copy sq_matrix_1 and sq_matrix_2 to device memory */
    cudaMalloc((void**) &sq_matrix_1_d, size);
    cudaMemcpy(sq_matrix_1_d, sq_matrix_1, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &sq_matrix_2_d, size);
    cudaMemcpy(sq_matrix_2_d, sq_matrix_2, size, cudaMemcpyHostToDevice);
    
    /*allocate sq_matrix_result on host */
    cudaMalloc((void**) &sq_matrix_result_d, size);
    
    /***************************************************
    Step 2: Invoke kernel 
    ****************************************************/
    short adjusted_block_width = 16;
    if (sq_dimension % 15 == 0) adjusted_block_width = 15;
    else if (sq_dimension % 14 == 0) adjusted_block_width = 14;
    else if (sq_dimension % 13 == 0) adjusted_block_width = 13;
    else if (sq_dimension % 12 == 0) adjusted_block_width = 12;
    else if (sq_dimension % 11 == 0) adjusted_block_width = 11;
    else if (sq_dimension % 10 == 0) adjusted_block_width = 10;
    else if (sq_dimension % 9 == 0) adjusted_block_width = 9;
    else if (sq_dimension % 8 == 0) adjusted_block_width = 8;
    else if (sq_dimension % 7 == 0) adjusted_block_width = 7;
    else if (sq_dimension % 6 == 0) adjusted_block_width = 6;
    else if (sq_dimension % 5 == 0) adjusted_block_width = 5;
    else if (sq_dimension % 4 == 0) adjusted_block_width = 4;
    else if (sq_dimension % 3 == 0) adjusted_block_width = 3;
    else if (sq_dimension % 2 == 0) adjusted_block_width = 2;
    else adjusted_block_width = 1;

    int blockNum = ceil(sq_dimension * 1.0 / adjusted_block_width);
    dim3 dimBlock(adjusted_block_width, adjusted_block_width);
    dim3 dimGrid(blockNum, blockNum);

    matrix_mul_kernel<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
    
    /***************************************************
    Step 3: Transfer result from device to host 
    ****************************************************/
    cudaMemcpy(sq_matrix_result, sq_matrix_result_d, size, cudaMemcpyDeviceToHost);
    cudaFree(sq_matrix_1_d);
    cudaFree(sq_matrix_2_d);
    cudaFree(sq_matrix_result_d);
  }  
} // namespace cuda
