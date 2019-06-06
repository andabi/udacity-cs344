/* Udacity HW5
 Histogramming for Speed

 The goal of this assignment is compute a histogram
 as fast as possible.  We have simplified the problem as much as
 possible to allow you to focus solely on the histogramming algorithm.

 The input values that you need to histogram are already the exact
 bins that need to be updated.  This is unlike in HW3 where you needed
 to compute the range of the data and then do:
 bin = (val - valMin) / valRange to determine the bin.

 Here the bin is just:
 bin = val

 so the serial histogram calculation looks like:
 for (i = 0; i < numElems; ++i)
 histo[val[i]]++;

 That's it!  Your job is to make it run as fast as possible!

 The values are normally distributed - you may take
 advantage of this fact in your implementation.

 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "utils.h"

__global__
void
histoSerial (const unsigned int* const d_vals, unsigned int* const d_histo,
	     int numVals)
{
  for (int i = 0; i < numVals; i++)
    {
      d_histo[d_vals[i]] += 1;
    }
}

__global__
void
histoAtomicAdd (const unsigned int* const d_vals, unsigned int* const d_histo,
		int numVals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= numVals)
    {
      return;
    }

  atomicAdd (&d_histo[d_vals[idx]], 1);
}

__global__
void
build_group (const unsigned int* const d_vals, unsigned int* const d_group,
	     int numVals, int binSize)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int bin_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx >= numVals || bin_idx >= binSize)
    {
      return;
    }

  int n_groups = (numVals + binSize - 1) / binSize;
  int bin = idx / n_groups;
//  printf("n_groups: %d, bin: %d\n", n_groups, bin);
  d_group[bin * numVals + idx] = (bin_idx == bin) ? d_vals[idx] : -1;
}

__global__
void
histoSharedMem (const unsigned int* const d_group, unsigned int* const d_histo,
		int numVals, int coarseBinSize, int numThreads)
{
  extern __shared__ unsigned int histo_sh[];

  int tx = threadIdx.x;
  int group_idx = blockIdx.x;

  // TODO out-of-range check

  int numValsPerThreads = (numVals + numThreads - 1) / numThreads;

  int idx = tx * numValsPerThreads;
  if (idx >= numVals) {
      return;
  }

  for (int i=idx; i<min(numVals, idx + numValsPerThreads); i++)
    {
       int val = d_group[group_idx * numVals + i];
       if (val >= 0)
	 {
	   atomicAdd(&histo_sh[val - group_idx * coarseBinSize], 1);
	 }
    }
  __syncthreads();

  if (tx == 0)
    {
      for (int i=0; i<coarseBinSize; i++)
	{
  	  d_histo[group_idx * coarseBinSize + i] = histo_sh[i];
	}
    }
}

void
computeHistogram (const unsigned int* const d_vals, //INPUT
    unsigned int* const d_histo,      //OUTPUT
    const unsigned int numBins, const unsigned int numElems)
{
  int dev = 0;
  cudaSetDevice (dev);

  cudaDeviceProp devProps;
  if (cudaGetDeviceProperties (&devProps, dev) == 0)
    {
      printf ("Using device %d:\n", dev);
      printf (
	  "%s; global mem: %luMB; shared mem: %luKB; max threads per block: %d\n",
	  devProps.name, devProps.totalGlobalMem / 1024 / 1024,
	  devProps.sharedMemPerBlock / 1024, devProps.maxThreadsPerBlock);
    }

  printf("# elems: %d\n", numElems);

  /* 1. serial version */
//  histoSerial <<<1, 1>>> (d_vals, d_histo, numElems);  // over 800 ms
  /* 2. using global memory and atomic add */
//  histoAtomicAdd <<<(numElems + 31) / 32, 32>>> (d_vals, d_histo, numElems); // around 3.1 ms

  /* TODO 3. using shared memory */
  const int COARSE_BIN_SIZE = 16;  // TODO experiment on the best size

  unsigned int* d_group;
  checkCudaErrors(
      cudaMalloc (&d_group, sizeof(unsigned int) * COARSE_BIN_SIZE * numElems));
  build_group<<<dim3((numElems + 31) / 32, 32), dim3(32, 1)>>> (d_vals, d_group, numElems, COARSE_BIN_SIZE);

  int n_groups = (numElems + COARSE_BIN_SIZE - 1) / COARSE_BIN_SIZE;
  histoSharedMem<<<n_groups, 32>>>(d_group, d_histo, numElems, COARSE_BIN_SIZE, 32);

  /* TODO 4. considering data dist (normal dist) */

  cudaDeviceSynchronize ();
  checkCudaErrors(cudaGetLastError ());
}
