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
  dprint (const T* const d_data, const size_t numElem)
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
  const unsigned int numBins = pow (2, bits);

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
	  assert(bin < numBins);
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
	     int numThreads, int numSegs)
{
  const int idx_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx_seg = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx_thread >= numThreads || idx_seg >= numSegs)
    {
      return;
    }

  const int idx = stride * idx_thread + (stride - 1);
  d_input[idx * numSegs + idx_seg] += d_input[(idx - stride / 2) * numSegs
      + idx_seg];
}

__global__ void
downswipe_scan (unsigned int* const d_input, unsigned int numElem, int stride,
		int numThreads, int numSegs)
{
  const int idx_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx_seg = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx_thread >= numThreads || idx_seg >= numSegs)
    {
      return;
    }

  const int inverse_idx_thread = numThreads - idx_thread - 1;
  const int idx = numElem - 2 * stride * inverse_idx_thread - 1;

  unsigned int left = d_input[(idx - stride) * numSegs + idx_seg];
  unsigned int right = d_input[idx * numSegs + idx_seg];

  d_input[(idx - stride) * numSegs + idx_seg] = right;
  d_input[idx * numSegs + idx_seg] = left + right;
}

void
segmented_exclusive_scan (unsigned int* const d_data, const size_t numElems,
			  const int numSegs = 1)
{
  assert((numElems & (numElems - 1)) == 0); //numElems must be power of 2;

// Blelloch exclusive scan
  int numSteps = log2 ((float) numElems);
  for (int i = 0; i < numSteps; i++)
    {
      int numThreads = pow (2, numSteps - i - 1);
      int stride = pow (2, i + 1);

      dim3 blockSize (32, 32);
      dim3 gridSize (numThreads / blockSize.x + 1, numSegs / blockSize.y + 1);
      reduce_scan <<<gridSize, blockSize>>> (d_data, numElems, stride,
					     numThreads, numSegs);
    }

  checkCudaErrors(
      cudaMemset (d_data + (numElems - 1) * numSegs, 0,
		  sizeof(unsigned int) * numSegs));

  for (int i = 0; i < numSteps; i++)
    {
      int numThreads = pow (2, i);
      int stride = pow (2, numSteps - i - 1);

      dim3 blockSize (32, 32);
      dim3 gridSize (numThreads / blockSize.x + 1, numSegs / blockSize.y + 1);
      downswipe_scan <<<gridSize, blockSize>>> (d_data, numElems, stride,
						numThreads, numSegs);
    }
}

void
exclusive_scan_sequantially (unsigned int* data, const size_t numElems)
{
  unsigned int scanned_data[numElems] =
    { };
    {
      for (size_t i = 1; i < numElems; i++)
	{
	  scanned_data[i] = scanned_data[i - 1] + data[i - 1];
	}
    }
  data = scanned_data;
}

void
__global__
segmented_mask (unsigned int* const d_inputVals, const unsigned int numElems,
		unsigned int* d_mask, const int numSegs, const int idx_iter,
		const int bits)
{
  const int idx_elem = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx_seg = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx_seg >= numSegs || idx_elem >= numElems)
    {
      return;
    }

  int bin = get_bin (d_inputVals[idx_elem], idx_iter, bits);
  if (bin == idx_seg)
    {
      d_mask[idx_elem * numSegs + idx_seg] = true;
    }
}

__global__ void
merge_segments (unsigned int* const d_input, unsigned int* const d_mask,
		const int numElems, unsigned int* const d_output,
		const int numSegs)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= numElems)
    {
      return;
    }

  int sum = 0;
  for (int i = 0; i < numSegs; i++)
    {
      sum += d_input[idx * numSegs + i] * d_mask[idx * numSegs + i];
    }
  d_output[idx] = sum;
}

void
get_absPos (unsigned int* const d_inputVals, const size_t numElems,
	    unsigned int* d_absPos, const size_t numBins, int idx_iter,
	    int bits)
{
  get_histogram (d_inputVals, numElems, d_absPos, idx_iter, bits);
  segmented_exclusive_scan (d_absPos, numBins, 1);
}

void
get_relPos (unsigned int* const d_inputVals, const size_t numElems,
	    unsigned int* d_relPos, const size_t numBins, int idx_iter,
	    int bits)
{
  const dim3 blockSize (32, 32);
  const dim3 gridSize (numElems / blockSize.x + 1, numBins / blockSize.y + 1);

  // build mask
  unsigned int* d_mask;
  checkCudaErrors(
      cudaMalloc (&d_mask, sizeof(unsigned int) * numElems * numBins));
  segmented_mask <<<gridSize, blockSize>>> (d_inputVals, numElems, d_mask,
					    numBins, idx_iter, bits);

  // segmented exclusive scan
  unsigned int* d_scan;
  int padding = pow (2, (int) (log2 ((float) numElems)) + 1) - numElems;
  int padded_numElems = padding + numElems;

  checkCudaErrors(
      cudaMalloc (&d_scan, sizeof(unsigned int) * padded_numElems * numBins));
  checkCudaErrors(
      cudaMemcpy (d_scan, d_mask, sizeof(unsigned int) * numElems * numBins,
		  cudaMemcpyDeviceToDevice));

  segmented_exclusive_scan (d_scan, padded_numElems, numBins);

  // merge segments
  merge_segments <<<numElems / 32 + 1, 32>>> (d_scan, d_mask, numElems,
					      d_relPos, numBins);
}

__global__ void
scatter (unsigned int* const d_inputVals, unsigned int* const d_inputPos,
	 const int numElems, int idx_iter, const int bits,
	 unsigned int* d_absPos, unsigned int* d_relPos,
	 unsigned int* const d_outputVals, unsigned int* const d_outputPos)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= numElems)
    {
      return;
    }

  int bin = get_bin (d_inputVals[idx], idx_iter, bits);
  int pos = d_absPos[bin] + d_relPos[idx];
  assert(pos < numElems);
  d_outputVals[pos] = d_inputVals[idx];
  d_outputPos[pos] = d_inputPos[idx];
}

void
your_sort (unsigned int* const d_inputVals, unsigned int* const d_inputPos,
	   unsigned int* const d_outputVals, unsigned int* const d_outputPos,
	   const size_t numElems)
{
  // TODO get absPos and get relPos could be done in parallel using Stream

  const int N_BITS = 4;  // power of 2
  const size_t numBins = pow (2, N_BITS);
  int numIters = CHAR_BIT * sizeof(unsigned int) / N_BITS;

  unsigned int *d_absPos, *d_relPos;
  checkCudaErrors(cudaMalloc (&d_absPos, sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMalloc (&d_relPos, sizeof(unsigned int) * numElems));

  unsigned int* t_inputVals = d_inputVals, *t_inputPos = d_inputPos,
      *t_outputVals = d_outputVals, *t_outputPos = d_outputPos;
  for (int i = 0; i < numIters; i++)
    {
      checkCudaErrors(cudaMemset (d_absPos, 0, sizeof(unsigned int) * numBins));
      get_absPos (t_inputVals, numElems, d_absPos, numBins, i, N_BITS);

      checkCudaErrors(cudaMemset (d_relPos, 0, sizeof(numElems)));
      get_relPos (t_inputVals, numElems, d_relPos, numBins, i, N_BITS);

      scatter <<<numElems / 32 + 1, 32>>> (t_inputVals, t_inputPos, numElems, i,
					   N_BITS, d_absPos, d_relPos,
					   t_outputVals, t_outputPos);

      swap(t_inputVals, t_outputVals);
      swap(t_inputPos, t_outputPos);
    }

  if (numIters % 2 == 0)
    {
      checkCudaErrors(
	  cudaMemcpy (d_outputVals, d_inputVals,
		      sizeof(unsigned int) * numElems,
		      cudaMemcpyDeviceToDevice));
      checkCudaErrors(
	  cudaMemcpy (d_outputPos, d_inputPos, sizeof(unsigned int) * numElems,
		      cudaMemcpyDeviceToDevice));
    }

  // cleanup
  checkCudaErrors(cudaFree (d_absPos));
  checkCudaErrors(cudaFree (d_relPos));

}
