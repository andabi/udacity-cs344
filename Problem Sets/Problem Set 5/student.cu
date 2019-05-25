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

#include "utils.h"

__global__
void
histoSerial (const unsigned int* const vals, unsigned int* const histo,
	     int numVals)
{
  for (int i = 0; i < numVals; i++)
    {
      histo[vals[i]] += 1;
    }
}

__global__
void
histoAtomicAdd (const unsigned int* const vals, unsigned int* const histo,
		int numVals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= numVals)
    {
      return;
    }

  atomicAdd (&histo[vals[idx]], 1);
}

void
computeHistogram (const unsigned int* const d_vals, //INPUT
    unsigned int* const d_histo,      //OUTPUT
    const unsigned int numBins, const unsigned int numElems)
{
  /* 1. serial version */
//  histoSerial <<<1, 1>>> (d_vals, d_histo, numElems);  // over 800 ms

  /* 2. using global memory and atomic add */
  dim3 block(32);
  dim3 grid(numElems / block.x + 1);
  histoAtomicAdd <<<grid, block>>> (d_vals, d_histo, numElems);  // around 3.1 ms

  /* TODO 3. using shared memory */

  /* TODO 4. considering data dist (normal dist) */

  cudaDeviceSynchronize ();
  checkCudaErrors(cudaGetLastError ());
}
