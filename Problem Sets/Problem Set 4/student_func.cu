//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
 ===============

 For this assignment we are implementing red eye removal.  This is
 accomplished by first creating a score for every pixel that tells us how
 likely it is to be a red eye pixel.  We have already done this for you - you
 are receiving the scores and need to sort them in ascending order so that we
 know which pixels to alter to remove the red eye.

 Note: ascending order == smallest to largest

 Each score is associated with a position, when you sort the scores, you must
 also move the positions accordingly.

 Implementing Parallel Radix Sort with CUDA
 ==========================================

 The basic idea is to construct a histogram on each pass of how many of each
 "digit" there are.   Then we scan this histogram so that we know where to put
 the output of each digit.  For example, the first 1 must come after all the
 0s so we have to know how many 0s there are to be able to start moving 1s
 into the correct position.

 1) Histogram of the number of occurrences of each digit
 2) Exclusive Prefix Sum of Histogram
 3) Determine relative offset of each digit
 For example [0 0 1 1 0 0 1]
 ->  [0 1 0 1 2 3 2]
 4) Combine the results of steps 2 & 3 to determine the final
 output location for each element and move it there

 LSB Radix sort is an out-of-place sort and you will need to ping-pong values
 between the input and output buffers we have provided.  Make sure the final
 sorted results end up in the output buffer!  Hint: You may need to do a copy
 at the end.

 */

using namespace std;

template<typename T>
  void
  print_device_data (const T* const d_data, const size_t numElem)
  {
    T *h_data = (T*) malloc (sizeof(T) * numElem);
    checkCudaErrors(
	cudaMemcpy (h_data, d_data, sizeof(T) * numElem,
		    cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < numElem; i++)
      {
	cout << h_data[i] << " ";
      }
    cout << endl;
  }

__host__ __device__ int
get_bin (unsigned int value, int idx_iter, const int bits)
{
  const int n_iter = sizeof(int) * CHAR_BIT / bits;
  const int bin = value << ((n_iter - idx_iter - 1) * bits)
      >> ((n_iter - 1) * bits);
  return bin;
}

__global__ void
histogram (unsigned int* const d_inputVals, const unsigned int numPixels,
	   unsigned int* const d_histogram, const int bits,
	   const unsigned int idx_iter)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= numPixels)
    {
      return;
    }

  const int bin = get_bin (d_inputVals[idx], idx_iter, bits);

  atomicAdd (&d_histogram[bin], 1);
}

unsigned int*
get_histogram (unsigned int* const d_inputVals, const size_t numPixels,
	       unsigned int* d_absPos, const size_t idx_iter, const int bits,
	       bool is_reference = false)
{
  const int numBins = pow (2, bits);

  const dim3 blockSize (32 * 32);
  const dim3 gridSize (numPixels / blockSize.x + 1);

  histogram <<<gridSize, blockSize>>> (d_inputVals, numPixels, d_absPos, bits,
				       idx_iter);

  if (is_reference)
    {
      size_t ref_histogram[numBins] =
	{ };
      unsigned int *h_histogram = (unsigned int*) malloc (
	  sizeof(unsigned int) * numBins);
      checkCudaErrors(
	  cudaMemcpy (h_histogram, d_absPos, sizeof(unsigned int) * numBins,
		      cudaMemcpyDeviceToHost));

      unsigned int *h_inputVals = (unsigned int*) malloc (
	  sizeof(unsigned int) * numPixels);
      checkCudaErrors(
	  cudaMemcpy (h_inputVals, d_inputVals,
		      sizeof(unsigned int) * numPixels,
		      cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < numPixels; i++)
	{
	  int bin = get_bin (h_inputVals[i], idx_iter, bits);
	  ref_histogram[bin] += 1;
	}
      cout << "(reference)" << endl;
      for (size_t i = 0; i < numBins; i++)
	{
	  cout << ref_histogram[i] << " (" << h_histogram[i] << "), ";
	}
      cout << endl;
    }

  return d_absPos;
}

__global__ void
init_scan (unsigned int* const d_histogram, unsigned int numBins,
	   unsigned int* const d_temp)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numBins)
    {
      return;
    }
  d_temp[idx] = d_histogram[idx];
}

__global__ void
copy_scan (unsigned int* const d_temp, unsigned int* const d_cdf,
	   unsigned int numBins)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numBins)
    {
      d_cdf[idx] = d_temp[idx];
    }
}

__global__ void
reduce_scan (unsigned int* const d_input, unsigned int numElem, int stride,
	     int numThreads)
{
  const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx >= numThreads)
    {
      return;
    }
  const int idx = stride * thread_idx + (stride - 1);
  d_input[idx] += d_input[idx - stride / 2];
}

__global__ void
downswipe_scan (unsigned int* const d_input, unsigned int numElem, int stride,
		int numThreads)
{
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx >= numThreads)
    {
      return;
    }
  thread_idx = numThreads - thread_idx - 1;
  const int idx = numElem - 2 * stride * thread_idx - 1;
  unsigned int left = d_input[idx - stride], right = d_input[idx];
  d_input[idx - stride] = right;
  d_input[idx] = left + right;
}

void
compute_exclusive_scan (unsigned int* const d_data, const size_t numBins)
{
//  // initialize d_cdf
//  dim3 blockSize (32);
//  dim3 gridSize (numBins / blockSize.x + 1);
//  init_scan <<<gridSize, blockSize>>> (d_histogram, numBins, d_cdf);

// Blelloch exclusive scan
  int numSteps = log2 ((float) numBins);
  for (int i = 0; i < numSteps; i++)
    {
      int numThreads = pow (2, numSteps - i - 1);
      int stride = pow (2, i + 1);
      reduce_scan <<<1, numThreads>>> (d_data, numBins, stride, numThreads);
    }
  checkCudaErrors(cudaMemset (&d_data[numBins - 1], 0, sizeof(unsigned int)));
  for (int i = 0; i < numSteps; i++)
    {
      int numThreads = pow (2, i);
      int stride = pow (2, numSteps - i - 1);
      downswipe_scan <<<1, numThreads>>> (d_data, numBins, stride, numThreads);
    }
}

void
compute_exclusive_scan_sequantially (unsigned int* data, const size_t numBins)
{
  unsigned int scaned_data[numBins] =
    { };
    {
      for (size_t i = 1; i < numBins; i++)
	{
	  scaned_data[i] = scaned_data[i - 1] + data[i - 1];
	}
    }
  data = scaned_data;
}

void
__global__
apply_mask (unsigned int* const d_inputVals, const unsigned int numElems,
	    bool* d_mask, const int numBins, const int idx_iter, const int bits)
{
  const int idx_elem = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx_bin = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx_bin >= numBins || idx_elem >= numElems)
    {
      return;
    }

  int bin = get_bin (d_inputVals[idx_elem], idx_iter, bits);
  if (bin == idx_bin)
    {
      d_mask[idx_elem * numBins + idx_bin] = true;
    }
}

void
get_absPos (unsigned int* const d_inputVals, const size_t numElems,
	    unsigned int* d_absPos, const int numBins, int idx_iter, int bits)
{
  get_histogram (d_inputVals, numElems, d_absPos, idx_iter, bits, false);
  compute_exclusive_scan (d_absPos, numBins);
}

void
get_relPos (unsigned int* const d_inputVals, const size_t numElems,
	    unsigned int* d_relPos, const int numBins, int idx_iter, int bits)
{
  bool* d_mask;
  checkCudaErrors(cudaMalloc (&d_mask, sizeof(bool) * numElems * numBins));
  const dim3 blockSize (32, 32);
  const dim3 gridSize (numElems / blockSize.x + 1, numBins / blockSize.y + 1);

  apply_mask <<<gridSize, blockSize>>> (d_inputVals, numElems, d_mask, numBins,
					idx_iter, bits);

  // segmented exclusive scan

  // sequaltially add
}

void
your_sort (unsigned int* const d_inputVals, unsigned int* const d_inputPos,
	   unsigned int* const d_outputVals, unsigned int* const d_outputPos,
	   const size_t numElems)
{
  const int N_BITS = 4;
  const int numBins = pow (2, N_BITS);

  int idx_iter = 0;

  unsigned int *d_absPos;
  checkCudaErrors(cudaMalloc (&d_absPos, sizeof(unsigned int) * numBins));
  get_absPos (d_inputVals, numElems, d_absPos, numBins, idx_iter, N_BITS);

  //// get rel pos
  unsigned int *d_relPos;
  checkCudaErrors(cudaMalloc (&d_relPos, sizeof(unsigned int) * numElems));
  get_relPos (d_inputVals, numElems, d_relPos, numBins, idx_iter, N_BITS);

  // scatter, iterations
  // make sure copy to output buffer

  // cleanup
  checkCudaErrors(cudaFree (d_absPos));

}
