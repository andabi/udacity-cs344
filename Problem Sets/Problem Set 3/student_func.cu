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

__global__ void
reduce_step (const float* const d_input, const int numPixels, float* const d_output,
	     const int stride, const int n_threads, const bool is_min)
{

  const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // out-of-bound check
  if (thread_idx >= n_threads)
    {
      return;
    }

  const int idx = thread_idx * stride;

//  printf("%d, %d, %d, %d\n", n_threads, idx, stride, numPixels);

  if (numPixels > idx + stride)
    {
      if (is_min) {
        d_output[idx] = min (d_input[idx], d_input[idx + stride]);
      } else {
        d_output[idx] = max (d_input[idx], d_input[idx + stride]);
      }
    }
  else
    {
      d_output[idx] = d_input[idx];
    }
}

void
reduce (const float* const d_input, const size_t numPixels, float* const d_output, const bool is_min)
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
      reduce_step <<<gridSize, blockSize>>> (d_temp, numPixels, d_output, stride,
					     n_threads, is_min);
      d_temp = d_output;
    }
}

void print_temp(const float* const d_data, const size_t numElem) {
  float *h_data = (float *) malloc(sizeof(float)*numElem);
  checkCudaErrors(cudaMemcpy(h_data, d_data, sizeof(float)*numElem, cudaMemcpyDeviceToHost));
  for (int i = 0; i < numElem; i++)
    {
      std::cout << h_data[i] << ",";
    }
}

void
get_range (const float* const d_logLuminance, const size_t numPixels, float &min_logLum,
	   float&max_logLum)
{
  float* d_output;
  checkCudaErrors(
      cudaMalloc (&d_output, sizeof(float) * numPixels));

  reduce (d_logLuminance, numPixels, d_output, true);
  checkCudaErrors(
      cudaMemcpy (&min_logLum, d_output, sizeof(float),
		  cudaMemcpyDeviceToHost));

  reduce (d_logLuminance, numPixels, d_output, false);
  checkCudaErrors(
      cudaMemcpy (&max_logLum, d_output, sizeof(float),
		  cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree (d_output));
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
//  const size_t numPixels = 10000;

  // find the min/max
  get_range (d_logLuminance, numPixels, min_logLum, max_logLum);

  print_temp(d_logLuminance, numPixels);

  printf ("min: %lf, max: %lf", min_logLum, max_logLum);
}
