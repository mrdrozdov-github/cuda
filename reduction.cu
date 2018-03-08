#include "optparse.h"

#include <cuda_runtime.h>
#include <math.h>
#include <cooperative_groups.h>

#include "common.h"


using namespace cooperative_groups;


const int BLOCK_SIZE = 512;


static int reduceCPU(int *data, int n)
{
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += data[i];
    return sum;
}

// Using cooperative groups

// 1. Interleaved Addressing (branch divergence)

__device__
void cg_reduce1_helper(thread_block block, int *idata, int *odata, int n, int height)
{
    __shared__ int cache[BLOCK_SIZE];
    int i = block.group_index().x * block.size() + block.thread_rank();
    int tid = block.thread_rank();
    cache[tid] = i < n ? idata[i] : 0;
    block.sync();

    for (unsigned int s = 1; s < block.size(); s*=2) {
        if (tid % (2*s) == 0) {
            cache[tid] += cache[tid + s];
        }
        block.sync();
    }

    if (tid == 0) odata[block.group_index().x] = cache[tid];
}

__global__
void cg_reduce1(int *idata, int *odata, int n, int height)
{
    cg_reduce1_helper(this_thread_block(), idata, odata, n, height);
}

// 2. Interleaved Addressing (bank conflicts)

__device__
void cg_reduce2_helper(thread_block block, int *idata, int *odata, int n, int height)
{
    __shared__ int cache[BLOCK_SIZE];
    int i = block.group_index().x * block.size() + block.thread_rank();
    int tid = block.thread_rank();
    cache[tid] = i < n ? idata[i] : 0;
    block.sync();

    for (unsigned int s = 1; s < block.size(); s*=2) {
        int index = 2 * s * tid;
        if (index < block.size()) {
            cache[index] += cache[index + s];
        }
        block.sync();
    }

    if (tid == 0) odata[block.group_index().x] = cache[tid];
}

__global__
void cg_reduce2(int *idata, int *odata, int n, int height)
{
    cg_reduce2_helper(this_thread_block(), idata, odata, n, height);
}

// 3. Sequential Addressing

__device__
void cg_reduce3_helper(thread_block block, int *idata, int *odata, int n, int height)
{
    __shared__ int cache[BLOCK_SIZE];
    int i = block.group_index().x * block.size() + block.thread_rank();
    int tid = block.thread_rank();
    cache[tid] = i < n ? idata[i] : 0;
    block.sync();

    for (unsigned int s = block.size()/2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        block.sync();
    }

    if (tid == 0) odata[block.group_index().x] = cache[tid];
}

__global__
void cg_reduce3(int *idata, int *odata, int n, int height)
{
    cg_reduce3_helper(this_thread_block(), idata, odata, n, height);
}

// 4. First Add During Load

__device__
void cg_reduce4_helper(thread_block block, int *idata, int *odata, int n, int height)
{
    __shared__ int cache[BLOCK_SIZE];
    int i = block.group_index().x * block.size() * 2 + block.thread_rank();
    int tid = block.thread_rank();
    cache[tid] = i < n ? idata[i] : 0;
    cache[tid] += i + block.size() < n ? idata[i+block.size()] : 0;
    block.sync();

    for (unsigned int s = block.size()/2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        block.sync();
    }

    if (tid == 0) odata[block.group_index().x] = cache[tid];
}

__global__
void cg_reduce4(int *idata, int *odata, int n, int height)
{
    cg_reduce4_helper(this_thread_block(), idata, odata, n, height);
}

// 5. Unroll the last Warp

__device__
void cg_reduce5_helper(thread_block block, int *idata, int *odata, int n, int height)
{
    __shared__ int cache[BLOCK_SIZE];
    int i = block.group_index().x * block.size() * 2 + block.thread_rank();
    int tid = block.thread_rank();
    cache[tid] = i < n ? idata[i] : 0;
    cache[tid] += i + block.size() < n ? idata[i+block.size()] : 0;
    block.sync();

    for (unsigned int s = block.size()/2; s > 32; s >>= 1) {
        if (tid < s)
            cache[tid] += cache[tid + s];
        block.sync();
    }

	if (tid < 32) {
        volatile int *vcache = cache;
		vcache[tid] += vcache[tid + 32];
		vcache[tid] += vcache[tid + 16];
		vcache[tid] += vcache[tid + 8];
		vcache[tid] += vcache[tid + 4];
		vcache[tid] += vcache[tid + 2];
		vcache[tid] += vcache[tid + 1];
	}

    if (tid == 0) odata[block.group_index().x] = cache[tid];
}

__global__
void cg_reduce5(int *idata, int *odata, int n, int height)
{
    cg_reduce5_helper(this_thread_block(), idata, odata, n, height);
}

// 6. Completely Unrolled

template <unsigned int blockSize>
__device__
void cg_reduce6_helper(thread_block block, int *idata, int *odata, int n, int height)
{
    __shared__ int cache[BLOCK_SIZE];
    int i = block.group_index().x * block.size() * 2 + block.thread_rank();
    int tid = block.thread_rank();
    cache[tid] = i < n ? idata[i] : 0;
    cache[tid] += i + block.size() < n ? idata[i+block.size()] : 0;
    block.sync();

    if (blockSize >= 512) {
        if (tid < 256) { cache[tid] += cache[tid + 256]; } block.sync();
    }
    if (blockSize >= 256) {
        if (tid < 128) { cache[tid] += cache[tid + 128]; } block.sync();
    }
    if (blockSize >= 128) {
        if (tid < 64) { cache[tid] += cache[tid + 64]; } block.sync();
    }

	if (tid < 32) {
        volatile int *vcache = cache;
		vcache[tid] += vcache[tid + 32];
		vcache[tid] += vcache[tid + 16];
		vcache[tid] += vcache[tid + 8];
		vcache[tid] += vcache[tid + 4];
		vcache[tid] += vcache[tid + 2];
		vcache[tid] += vcache[tid + 1];
	}

    if (tid == 0) odata[block.group_index().x] = cache[tid];
}

template <unsigned int blockSize>
__global__
void cg_reduce6(int *idata, int *odata, int n, int height)
{
    cg_reduce6_helper<blockSize>(this_thread_block(), idata, odata, n, height);
}

// 7. Warp Shuffle

__inline__ __device__
int warpReduceSum(int val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

__inline__ __device__
int blockReduceSum(int val) {

  static __shared__ int shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__device__
void deviceReduceKernel(int *in, int* out, int N) {
  int sum = 0;
  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x==0)
    out[blockIdx.x]=sum;
}

__global__
void cg_reduce7(int *idata, int *odata, int n, int height)
{
	deviceReduceKernel(idata, odata, n);
}

// 8. Warp Shuffle w. AtomicAdd

__device__
void deviceReduceWarpAtomicKernel(int *in, int* out, int N) {
  int sum = int(0);
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
      i < N; 
      i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = warpReduceSum(sum);
  if ((threadIdx.x & (warpSize - 1)) == 0)
    atomicAdd(out, sum);
}

__global__
void cg_reduce8(int *idata, int *odata, int n, int height)
{
	deviceReduceWarpAtomicKernel(idata, odata, n);
}

// TODO put your kernel here
__global__
void reduceGPU(int *idata, int *odata, int n)
{
    __shared__ int cache[BLOCK_SIZE];
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;
    int localDim = blockDim.x;
    int outputIdx = blockIdx.x;
    if (globalIdx < n) {
        cache[localIdx] = idata[globalIdx];
    } else {
        cache[localIdx] = 0;
    }

    __syncthreads();

    if (localIdx == 0) {
        int sum = 0;
        for (int i = 0; i < localDim; i++) {
            sum += cache[i];
        }
        odata[outputIdx] = sum;
    }
}


__global__
void reduceGPULoop(int *idata, int *odata, int n, int height)
{
    __shared__ int cache[BLOCK_SIZE];
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;
    int outputIdx = blockIdx.x;
    if (globalIdx < n) {
        cache[localIdx] = idata[globalIdx];
    } else {
        printf("miss %d\n", globalIdx);
        cache[localIdx] = 0;
    }

    __syncthreads();

    // log2(blocksize) iterations
    for (int i = 0; i < height; i++) {
        if (localIdx % (int) pow(2., i+1) == 0) {
            int offset = pow(2., i);
            cache[localIdx] += cache[localIdx + offset];
        }

        __syncthreads();
    }

    if (localIdx == 0) {
        odata[outputIdx] = cache[localIdx];
    }
}


__global__
void reduceGPULoopBankConflict(int *idata, int *odata, int n, int height)
{
    __shared__ int cache[BLOCK_SIZE];
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;
    int outputIdx = blockIdx.x;
    if (globalIdx < n) {
        cache[localIdx] = idata[globalIdx];
    } else {
        cache[localIdx] = 0;
    }

    __syncthreads();

    // log2(blocksize) iterations
    for (int i = 0; i < height; i++) {
        int offset = BLOCK_SIZE / (int) pow(2., i+1);
        if (localIdx < offset) {
            cache[localIdx] += cache[localIdx + offset];
        }
        __syncthreads();
    }

    if (localIdx == 0) {
        odata[outputIdx] = cache[localIdx];
    }
}


int main(int argc, char *argv[])
{
    optparse::OptionParser parser = optparse::OptionParser();

    char const *const choices[] = {"cg1", "cg2", "cg3", "cg4", "cg5", "cg6", "cg7", "cg8"};
    parser.add_option("-m", "--method").choices(&choices[0], &choices[8]).set_default("cg1");

    const optparse::Values &options = parser.parse_args(argc, argv);
    const std::vector<std::string> args = parser.args();

    std::cout << "method: " << options["method"] << std::endl;

    int n = (4 << 20);  // number of elements to reduce

    unsigned bytes = n * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    for (int i=0; i<n; i++)
        h_idata[i] = (int)(rand() & 0xFF);

    // TODO determine numBlocks and numThreads
    int numThreads = BLOCK_SIZE;
    int numBlocks = ceil((float)n/(float)numThreads);
    int height = log2((float)numThreads);
    printf("n=%d nThreads=%d nBlocks=%d height=%d\n", n, numThreads, numBlocks, height);

    // allocate device memory and data
    int *d_idata = NULL, *d_odata = NULL;
    CUDA(cudaMalloc((void **) &d_idata, bytes));
    CUDA(cudaMalloc((void **) &d_odata, numBlocks*sizeof(int)));  // FIX
    // copy data to device memory
    CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    int gpu_result = 0;
    cudaEvent_t start, stop;
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&stop));

    // Start record
    CUDA(cudaEventRecord(start, 0));

    // TODO call your reduce kernel(s) with the right parameters
    // INPUT:       d_idata
    // OUTPUT:      d_odata
    // ELEMENTS:    n

    // (1) reduce across all elements
    std::string method = options["method"];
    if (strcmp(method.c_str(), choices[0]) == 0) {
        cg_reduce1<<<numBlocks, numThreads>>>(d_idata, d_odata, n, height);
    } else if (strcmp(method.c_str(), choices[1]) == 0) {
        cg_reduce2<<<numBlocks, numThreads>>>(d_idata, d_odata, n, height);
    } else if (strcmp(method.c_str(), choices[2]) == 0) {
        cg_reduce3<<<numBlocks, numThreads>>>(d_idata, d_odata, n, height);
    } else if (strcmp(method.c_str(), choices[3]) == 0) {
        numBlocks /= 2;
        cg_reduce4<<<numBlocks, numThreads>>>(d_idata, d_odata, n, height);
    } else if (strcmp(method.c_str(), choices[4]) == 0) {
        numBlocks /= 2;
        cg_reduce5<<<numBlocks, numThreads>>>(d_idata, d_odata, n, height);
    } else if (strcmp(method.c_str(), choices[5]) == 0) {
        numBlocks /= 2;
        cg_reduce6<BLOCK_SIZE><<<numBlocks, numThreads>>>(d_idata, d_odata, n, height);
    } else if (strcmp(method.c_str(), choices[6]) == 0) {
        numBlocks /= 2;
        cg_reduce7<<<numBlocks, numThreads>>>(d_idata, d_odata, n, height);
    } else if (strcmp(method.c_str(), choices[7]) == 0) {
        numBlocks /= 2;
        cg_reduce8<<<numBlocks, numThreads>>>(d_idata, d_odata, n, height);
    } else {
        printf("invalid method: %s\n", method.c_str());
    }
    // reduceGPULoop<<<numBlocks, numThreads>>>(d_idata, d_odata, n, height);
    // reduceGPULoopBankConflict<<<numBlocks, numThreads>>>(d_idata, d_odata, n, height);

    // (2) reduce across all blocks
    size_t block_bytes = numBlocks * sizeof(int);
    int *h_blocks = (int *)malloc(block_bytes);
    CUDA(cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < numBlocks; ++i)
        gpu_result += h_blocks[i];

    CUDA(cudaEventRecord(stop, 0));
    CUDA(cudaEventSynchronize(stop));
    float time_ms;
    CUDA(cudaEventElapsedTime(&time_ms, start, stop)); // that's the time your kernel took to run in ms!
    printf("bandwidth %.2f GB/s   elements %u   blocks %u   threads %u\n",
           1e-9 * bytes/(time_ms/1e3), n, numBlocks, numThreads);

    // check result against CPU
    int cpu_result = reduceCPU(h_idata, n);
    printf("gpu %u   cpu %u   ", gpu_result, cpu_result);
    printf((gpu_result==cpu_result) ? "pass\n" : "FAIL\n");

    // cleanup
    CUDA(cudaEventDestroy(start));
    CUDA(cudaEventDestroy(stop));
    free(h_idata);
    CUDA(cudaFree(d_idata));
    CUDA(cudaFree(d_odata));

    return 0;
}
