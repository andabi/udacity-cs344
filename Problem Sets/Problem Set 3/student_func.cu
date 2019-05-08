/* Udacity Homework 3
 HDR Tone-mapping

 Background HDR
 ==============

 A High Dynamic Range (HDR) image contains a wider variation of intensity
 and color than is allowed by the RGB format with 1 byte per channel that we
 have used in the previous assignment.

 To store this extra information we use single precision floating point for
 each channel.  This allows for an extremely wide range of intensity values.

 In the image for this assignment, the inside of church with light coming in
 through stained glass windows, the raw input floating point values for the
 channels range from 0 to 275.  But the mean is .41 and 98% of the values are
 less than 3!  This means that certain areas (the windows) are extremely bright
 compared to everywhere else.  If we linearly map this [0-275] range into the
 [0-255] range that we have been using then most values will be mapped to zero!
 The only thing we will be able to see are the very brightest areas - the
 windows - everything else will appear pitch black.

 The problem is that although we have cameras capable of recording the wide
 range of intensity that exists in the real world our monitors are not capable
 of displaying them.  Our eyes are also quite capable of observing a much wider
 range of intensities than our image formats / monitors are capable of
 displaying.

 Tone-mapping is a process that transforms the intensities in the image so that
 the brightest values aren't nearly so far away from the mean.  That way when
 we transform the values into [0-255] we can actually see the entire image.
 There are many ways to perform this process and it is as much an art as a
 science - there is no single "right" answer.  In this homework we will
 implement one possible technique.

 Background Chrominance-Luminance
 ================================

 The RGB space that we have been using to represent images can be thought of as
 one possible set of axes spanning a three dimensional space of color.  We
 sometimes choose other axes to represent this space because they make certain
 operations more convenient.

 Another possible way of representing a color image is to separate the color
 information (chromaticity) from the brightness information.  There are
 multiple different methods for doing this - a common one during the analog
 television days was known as Chrominance-Luminance or YUV.

 We choose to represent the image in this way so that we can remap only the
 intensity channel and then recombine the new intensity values with the color
 information to form the final image.

 Old TV signals used to be transmitted in this way so that black & white
 televisions could display the luminance channel while color televisions would
 display all three of the channels.


 Tone-mapping
 ============

 In this assignment we are going to transform the luminance channel (actually
 the log of the luminance, but this is unimportant for the parts of the
 algorithm that you will be implementing) by compressing its range to [0, 1].
 To do this we need the cumulative distribution of the luminance values.

 Example
 -------

 input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
 min / max / range: 0 / 9 / 9

 histo with 3 bins: [4 7 3]

 cdf : [4 11 14]


 Your task is to calculate this cumulative distribution by following these
 steps.

 */

#include "utils.h"
#include "stdio.h"

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

__global__ void
reduce_step (const float* const d_input, const int numPixels,
	     float* const d_output, const int stride, const int n_threads,
	     const bool is_min)
{

  const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // out-of-bound check
  if (thread_idx >= n_threads)
    {
      return;
    }

  const int idx = thread_idx * 2 * stride;

//  printf("%d, %d, %d, %d\n", n_threads, idx, stride, numPixels);

  if (numPixels > idx + stride)
    {
      if (is_min)
	{
	  d_output[idx] = min (d_input[idx], d_input[idx + stride]);
	}
      else
	{
	  d_output[idx] = max (d_input[idx], d_input[idx + stride]);
	}
    }
  else
    {
      d_output[idx] = d_input[idx];
    }
}

void
reduce_sequential (const float* const h_input, const size_t numPixels,
		   float &h_output, const bool is_min)
{
  assert(numPixels > 0);

  h_output = h_input[0];
  for (size_t i = 1; i < numPixels; i++)
    {
      if (is_min)
	{
	  h_output = min (h_output, h_input[i]);
	}
      else
	{
	  h_output = max (h_output, h_input[i]);
	}
    }
}

void
reduce (const float* const d_input, const size_t numPixels,
	float* const d_output, const bool is_min)
{
  assert(numPixels > 0);

  const int n_steps = ceil (log2 ((float) numPixels));
  const float* d_temp = d_input;

  for (int i = 0; i < n_steps; i++)
    {
      int n_threads = ceil (numPixels / pow (2, i + 1));

      const dim3 blockSize (32);
      dim3 gridSize (n_threads / blockSize.x + 1);

      const int stride = pow (2, i);
      reduce_step <<<gridSize, blockSize>>> (d_temp, numPixels, d_output,
					     stride, n_threads, is_min);
      d_temp = d_output;
    }
}

void
find_range (const float* const d_logLuminance, const size_t numPixels,
	    float &min_logLum, float &max_logLum, bool is_reference = false)
{
  float* d_output;
  checkCudaErrors(cudaMalloc (&d_output, sizeof(float) * numPixels / 2));
  reduce (d_logLuminance, numPixels, d_output, true);
  checkCudaErrors(
      cudaMemcpy (&min_logLum, d_output, sizeof(float),
		  cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree (d_output));

  checkCudaErrors(cudaMalloc (&d_output, sizeof(float) * numPixels / 2));
  reduce (d_logLuminance, numPixels, d_output, false);
  checkCudaErrors(
      cudaMemcpy (&max_logLum, d_output, sizeof(float),
		  cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree (d_output));

//  cout << "min: " << min_logLum << ", max: " << max_logLum << endl;

  if (is_reference)
    {
      float *h_logLuminance = (float *) malloc (sizeof(float) * numPixels);
      float h_output_max, h_output_min;
      checkCudaErrors(
	  cudaMemcpy (h_logLuminance, d_logLuminance, sizeof(float) * numPixels,
		      cudaMemcpyDeviceToHost));

      reduce_sequential (h_logLuminance, numPixels, h_output_min, true);
      reduce_sequential (h_logLuminance, numPixels, h_output_max, false);

      cout << "(reference) min: " << h_output_min << ", max: " << h_output_max
	  << endl;
    }
}

__global__ void
histogram (const float* const d_logLuminance, unsigned int* const d_histogram,
	   const int numPixels, const float min_logLum, const float max_logLum,
	   const int numBins)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= numPixels)
    {
      return;
    }

  const int bin = (d_logLuminance[idx] - min_logLum) / (max_logLum - min_logLum)
      * numBins;
  atomicAdd (&d_histogram[bin], 1);
}

unsigned int*
get_histogram (const float* const d_logLuminance, const size_t numPixels,
	       const float min_logLum, const float max_logLum,
	       const size_t numBins, bool is_reference = false)
{
  unsigned int* d_histogram;
  checkCudaErrors(cudaMalloc (&d_histogram, sizeof(unsigned int) * numBins));

  const dim3 blockSize (32 * 32);
  const dim3 gridSize (numPixels / blockSize.x + 1);

  histogram <<<gridSize, blockSize>>> (d_logLuminance, d_histogram, numPixels,
				       min_logLum, max_logLum, numBins);

  if (is_reference)
    {
      size_t ref_histogram[numBins] =
	{ };
      float *h_logLuminance = (float*) malloc (sizeof(float) * numPixels);

      unsigned int *h_histogram = (unsigned int*) malloc (
	  sizeof(unsigned int) * numBins);
      checkCudaErrors(
	  cudaMemcpy (h_histogram, d_histogram, sizeof(unsigned int) * numBins,
		      cudaMemcpyDeviceToHost));

      checkCudaErrors(
	  cudaMemcpy (h_logLuminance, d_logLuminance, sizeof(float) * numPixels,
		      cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < numPixels; i++)
	{
	  int bin = (h_logLuminance[i] - min_logLum) / (max_logLum - min_logLum)
	      * numBins;
	  ref_histogram[bin] += 1;
	}
      cout << "(reference)" << endl;
      for (size_t i = 0; i < numBins; i++)
	{
	  cout << ref_histogram[i] << " (" << h_histogram[i] << "), ";
	}
      cout << endl;
    }

  return d_histogram;
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
compute_exclusive_scan (unsigned int* const d_histogram, const size_t numBins,
			unsigned int* const d_cdf, bool is_reference)
{
  // initialize d_cdf
  dim3 blockSize (32);
  dim3 gridSize (numBins / blockSize.x + 1);
  init_scan <<<gridSize, blockSize>>> (d_histogram, numBins, d_cdf);

  // Blelloch exclusive scan
  int numSteps = log2 ((float) numBins);
  for (int i = 0; i < numSteps; i++)
    {
      int numThreads = pow (2, numSteps - i - 1);
      int stride = pow (2, i + 1);
      reduce_scan <<<1, numThreads>>> (d_cdf, numBins, stride, numThreads);
    }
  checkCudaErrors(cudaMemset (&d_cdf[numBins - 1], 0, sizeof(unsigned int)));
  for (int i = 0; i < numSteps; i++)
    {
      int numThreads = pow (2, i);
      int stride = pow (2, numSteps - i - 1);
      downswipe_scan <<<1, numThreads>>> (d_cdf, numBins, stride, numThreads);
    }

  if (is_reference)
    {
      unsigned int* ref_histogram = (unsigned int*) malloc (
	  sizeof(unsigned int) * numBins);
      checkCudaErrors(
	  cudaMemcpy (ref_histogram, d_histogram,
		      sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

      unsigned int ref_exclusive_scan[numBins] =
	{ };
      unsigned int sum = 0;
      cout << "(reference)" << endl;
      for (size_t i = 0; i < numBins; i++)
	{
	  ref_exclusive_scan[i] = sum;
	  sum += ref_histogram[i];
	  cout << ref_exclusive_scan[i] << " ";
	}
      cout << endl;
    }
}

void
your_histogram_and_prefixsum (const float* const d_logLuminance,
			      unsigned int* const d_cdf, float &min_logLum,
			      float &max_logLum, const size_t numRows,
			      const size_t numCols, const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
   1) find the minimum and maximum value in the input logLuminance channel
   store in min_logLum and max_logLum
   2) subtract them to find the range
   3) generate a histogram of all the values in the logLuminance channel using
   the formula: bin = (lum[i] - lumMin) / lumRange * numBins
   4) Perform an exclusive scan (prefix sum) on the histogram to get
   the cumulative distribution of luminance values (this should go in the
   incoming d_cdf pointer which already has been allocated for you)       */

  const size_t numPixels = numRows * numCols;

  // find the min/max
  find_range (d_logLuminance, numPixels, min_logLum, max_logLum);

// get historgram
  unsigned int* const d_histogram = get_histogram (d_logLuminance, numPixels,
						   min_logLum, max_logLum,
						   numBins, false);
// exclusive scan
  compute_exclusive_scan (d_histogram, numBins, d_cdf, false);
//  print_device_data<unsigned int> (d_cdf, numBins);

// cleanup
  checkCudaErrors(cudaFree (d_histogram));
}

