#include "optparse.h"

#include <iostream>
#include <vector>
#include <stdio.h>
#include <string>

using std::vector;
using namespace std;


// parameter describing the size of matrix A
const int rows = 20;
const int cols = 20;

const int BLOCK_SIZE = 2;

// TODO: transpose kernel
__global__
void transpose(float* a, float *b) {
    // calculate global id
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	b[col*rows + row] = a[row*cols + col];
}

__global__
void transpose_shared(float* a, float *b) {
    __shared__ float cache[BLOCK_SIZE][BLOCK_SIZE];

    // calculate global id
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

    cache[threadIdx.x][threadIdx.y] = a[row*cols + col];
    __syncthreads();

	row = blockDim.y * blockIdx.y + threadIdx.x;
	col = blockDim.x * blockIdx.x + threadIdx.y;

    b[col*cols + row] = cache[threadIdx.y][threadIdx.x];
}

void print_matrix(float *m, bool transposed) {
    int I = transposed ? cols : rows;
    int J = transposed ? rows : cols;
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            int key;
            if (transposed) {
                key = i * rows + j;
            } else {
                key = i * cols + j;
            }
            printf("%.1f ", m[key]);
        }
        printf("\n");
    }
    printf("done\n");
}

bool check_matrix(float *a, float *b, int rows, int cols, bool transposed) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int aIdx = i * rows + j;
            int bIdx = transposed ? j * rows + i : aIdx;

            if (a[aIdx] != b[bIdx]) {
                printf("a[%d] = %f does not equal b[%d] = %f\n", aIdx, a[aIdx], bIdx, b[bIdx]);
                return false;
            }
        }
    }
    return true;
}

// the main program starts life on the CPU and calls device kernels as required
int main(int argc, char *argv[])
{

    // allocate space in the host for storing input arrays (a and b) and the output array (c)
    size_t size = rows * cols * sizeof(float);
    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);

    // define device pointers for the same arrays when they'll be copied to the device
    float *_a, *_b;

    // allocate memory on the device (GPU) and check for errors (if any) during this call
    cudaError_t err;

    // allocate space for matrix A 
    err = cudaMalloc((void **) &_a, rows*cols*sizeof(float));
    if (err!= cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // allocate space for matrix B
    err = cudaMalloc((void **) &_b, rows*cols*sizeof(float));
    if (err!= cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Fill matrix A
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int key = i * cols + j;
            a[key] = i;
        }
    }

    print_matrix(a, false);

    // Copy array contents of A from the host (CPU) to the device (GPU)
    // Note that this is copied to the "global" memory on the device and is accessible to all threads in all blocks
    cudaMemcpy(_a, a, rows*cols*sizeof(float), cudaMemcpyHostToDevice);

    // assign a 2D distribution of 16 x 16 CUDA "threads" within each CUDA "block"
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    // calculate number of blocks along X and Y in a 2D CUDA "grid"
    dim3 numBlocks(
            ceil(float(rows)/float(threadsPerBlock.x)),
            ceil(float(cols)/float(threadsPerBlock.y))
            );

    float time;

    // create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord( start, 0);

    // launch the GPU kernel
	//transpose<<<numBlocks, threadsPerBlock>>>(_a, _b);
	transpose_shared<<<numBlocks, threadsPerBlock>>>(_a, _b);

    // stop the timer
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop);

    // print out the time required for the kernel to finish the transpose operation
    double Bandwidth = 2.0*1000*(double)(rows*cols*sizeof(float)) / (1000*1000*1000*time);
    std::cout << "Elapsed Time  = " << time << " Bandwidth used (GB/s) = " << Bandwidth << std::endl;

    // copy the answer back to the host (CPU) from the device (GPU)
    cudaMemcpy(b, _b, cols*rows*sizeof(float), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(_a);
    cudaFree(_b);

    // print matrix
    print_matrix(b, true);

    // check matrix values
    printf("Matrix a is transpose of matrix b: %d\n",
            check_matrix(a, b, rows, cols, true));

    // successful program termination
    return 0;
}
