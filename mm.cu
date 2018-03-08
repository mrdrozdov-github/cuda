/*
 * CUDA C++ code to multiply two square matrices
 *
 * To compile and link this example, use
 *
 *     nvcc matMul.cu -o matMul.x
 *
 * To run this code, use
 *
 *     ./matMul.x
 */

#include <iostream>

// parameter describing the size of the matrices
const int rows = 1024;
const int cols = 1024;

// block size for tiled multiplication using shared memory 
const int BLOCK_SIZE = 16;

// total number of blocks along X and Y
const int NUM_BLOCKS = rows/BLOCK_SIZE;

// print the matrix
void displayMatrix(float *a)
{
    std::cout << std::endl;
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            std::cout << a[i*cols+j] << " ";
        }
        std::cout << std::endl;
    }
}

// using global memory
__global__ void matrixMultiplyNaive(float *_a,   // pointer to matrix A on the device
                                    float *_b,   // pointer to matrix B on the device
                                    float *_c)   // pointer to matrix C = AB on the device
{
// TODO: Add the calculation of the inner product using global memory
}

__global__ void matrixMultiplyTiled(float *_a,   // pointer to matrix A on the device
                                    float *_b,   // pointer to matrix B on the device
                                    float *_c)   // pointer to matrix C = AB on the device
{
// TODO: calculate the thread mapping to the product matrix element
// TODO: define two shared memory arrays, one for matrix A and one for matrix B
// TODO: Add in a loop over the tiles in A and B
// TODO: calculate the partial inner products and add them up to get the element of the product matrix
}

// the main program starts life on the CPU and calls device kernels as required
int main(int argc, char *argv[])
{
    // allocate space in the host for storing input arrays (a and b) and the output array (c)
    float *a = new float[rows*cols];
    float *b = new float[rows*cols];
    float *c = new float[rows*cols];

    // define device pointers for the same arrays when they'll be copied to the device
    float *_a, *_b, *_c;

    // allocate memory on the device (GPU) and check for errors (if any) during this call
    cudaError_t err;

    // allocate space for matrix A 
    err = cudaMalloc((void **) &_a, rows*cols*sizeof(float));
    if (err!= cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }

    // allocate space for matrix B
    err = cudaMalloc((void **) &_b, rows*cols*sizeof(float));
    if (err!= cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }

    // allocate space for matrix C = AB
    err = cudaMalloc((void **) &_c, rows*cols*sizeof(float));
    if (err!= cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }

    // Fill matrix A
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            a[row + col*rows] = 2.0;
        }
    }
    if((rows<33) && (cols<33)) displayMatrix(a);

    // Fill matrix B
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            b[row + col*rows] = 4.0;
        }
    }
    if((rows<33) && (cols<33)) displayMatrix(b);

    // Copy array contents of A and B from the host (CPU) to the device (GPU)
    // Note that this is copied to the "global" memory on the device and is accessible to all threads in all blocks
    // WARNING: Global memory is slow (latency of a few 100 cycles)
    // 
    cudaMemcpy(_a, a, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b, rows*cols*sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    
    // calculate number of blocks along X and Y in a 2D CUDA "grid"
    dim3 dimGrid( ceil(float(cols)/float(dimBlock.x)), ceil(float(rows)/float(dimBlock.y)), 1 );

    float time;

    // create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord( start, 0);

    // launch the GPU kernel for parallel matrix multiplication of A and B
//  matrixMultiplyNaive<<<dimGrid,dimBlock>>>(_a, _b, _c);

    matrixMultiplyTiled<<<dimGrid,dimBlock>>>(_a, _b, _c);

    // stop the timer
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop);

    // print out the number of GFLOPs
    double GFLOPs = (double)(rows*cols) * 2*rows * 1000 / (1000*1000*1000*time);
    std::cout << "Elapsed Time  = " << time << " GFLOPs = " << GFLOPs << std::endl;

    // copy the answer back to the host (CPU) from the device (GPU)
    cudaMemcpy(c, _c, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

    if((rows<33) && (cols<33)) displayMatrix(c);

    // free device memory
    cudaFree(_a);
    cudaFree(_b);
    cudaFree(_c);

    // free host memory
    delete a;
    delete b;
    delete c;

    // successful program termination
    return 0;
}
