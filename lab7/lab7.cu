// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here


__global__ void gray_scale(float *input, unsigned char *output, unsigned char*output_2, int width, int height){
 
  // __shared__  char array_color[channels];

  int index = blockDim.x*blockIdx.x +threadIdx.x;
  if((index*3 +2)<(width*height*3)){
    // do cast inside gray scale
    output_2[index*3] = (unsigned char)(255*input[index*3]);
    output_2[(index*3)+1] = (unsigned char)(255*input[index*3 + 1]);
    output_2[(index*3)+2] = (unsigned char)(255*input[index*3 + 2]);
  } 
  if(index<(width*height)){
    unsigned char r = (unsigned char)(255*input[index*3]);
    unsigned char g = (unsigned char)(255*input[index*3 + 1]);
    unsigned char b = (unsigned char)(255*input[index*3 + 2]);
    output[index] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }

}


__global__ void comp_histo(unsigned char *input, unsigned int *output, int width, int height){

  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
  if(threadIdx.x < (HISTOGRAM_LENGTH)){
    histo_private[threadIdx.x]= 0;
  }

  __syncthreads();

  int i = blockIdx.x * blockDim.x +threadIdx.x;
  int stride = blockDim.x*gridDim.x;

  while(i<(width*height)){
    atomicAdd(&(histo_private[input[i]]), 1);
    i+=stride;
  }
  __syncthreads();

  if(threadIdx.x<256){
    atomicAdd(&(output[threadIdx.x]), histo_private[threadIdx.x]);
  }
}


__global__ void cummulative_dis(unsigned int *input, float *output, int width, int height, int len){

  // use lab6 code but for one block so no auxilary array
  __shared__ float partialSum[256*2];
  int cur_index = blockIdx.x*blockDim.x +threadIdx.x;


  partialSum[threadIdx.x] = float(input[cur_index]);
  
  
  //not claculating two according to daksh
  // if(2*blockIdx.x*blockDim.x+blockDim.x + t <len){
  // partialSum[t+blockDim.x] = float(input[2*blockIdx.x*blockDim.x+blockDim.x+t]/(width*height));
  // }else{
  // partialSum[t+blockDim.x] = 0;
  // }
  //  __syncthreads();
  

  // scan/post scan
  int stride = 1;
  while(stride<256){
      __syncthreads();
      int index = ((threadIdx.x+1)*stride*2)-1;
    if(index < 256 && (index-stride) >=0){
      partialSum[index] += partialSum[index-stride];
    }
      stride = stride*2;
    }

  stride = 256/2;
  while(stride>0){
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 -1;
    if((index+stride)<256){
      partialSum[index+stride] += partialSum[index];
    }
    stride = stride/2;
  }
  __syncthreads();


  if(cur_index < len){ // update cdf which is our output taking all elements not just final sum
    output[cur_index] = partialSum[threadIdx.x]/(width*height);
  }

}


__global__ void hist_equal (unsigned char * input, float* output, float* cdf, int width, int height){
  int index = blockIdx.x*blockDim.x +threadIdx.x;
  float cdfmin = cdf[0];

  if(index < width*height*3){

    float new_val = float(((cdf[input[index]]-cdfmin)/(1-cdfmin)));  // correct color step but don't convert to char like in git hub
    // why the fuck min max not work
    if(new_val>0.0){
      if(new_val<255.0){
        output[index] = new_val;
      }
      else{
        output[index] = 255.0;
      }
      __syncthreads();
    }
    else{
      output[index] = 0.0;
    }
  }
  // can skip casting char back to float
}





int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  
 // my added stuff
  float * deviceInput;
  unsigned char *input_c;
  unsigned char *gray;
  unsigned int *hist;
  float *cdf;
  float *deviceoutput;



  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);


  // get data even tho output pointless
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);






  // malloc my shit
  cudaMalloc((void**) &deviceInput, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**) &input_c, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void**) &gray, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void**) &hist, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void**) &cdf, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void**) &deviceoutput, imageWidth * imageHeight * imageChannels * sizeof(float));


  // set hist to zero so we dont get weird stuff
  cudaMemset((void *) hist, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset((void *) cdf, 0, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemcpy(deviceInput, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);


  // start kernels

  // make size without rgb
  dim3 dimBlock1(HISTOGRAM_LENGTH,1,1);
  dim3 dimGrid1(ceil((1.0*imageWidth*imageHeight)/(HISTOGRAM_LENGTH)),1,1);
  gray_scale<<<dimGrid1, dimBlock1>>>(deviceInput, gray, input_c, imageWidth, imageHeight);
  comp_histo<<<dimGrid1, dimBlock1>>>(gray, hist, imageWidth, imageHeight);
 
 // make histogram length number of threads for scan of one block
  dim3 dimBlock2(HISTOGRAM_LENGTH,1,1);
  dim3 dimGrid2(1,1,1);
  cummulative_dis<<<dimGrid2, dimBlock2>>>(hist, cdf, imageWidth, imageHeight, 256);

  // make size of full thing since I am copying over not gray but translated char which happened in gray

  dim3 dimBlock3(HISTOGRAM_LENGTH,1,1);
  dim3 dimGrid3(ceil((1.0*imageWidth*imageHeight*imageChannels)/(HISTOGRAM_LENGTH)),1,1);
  hist_equal<<<dimGrid3, dimBlock3>>>(input_c, deviceoutput, cdf, imageWidth, imageHeight);


  cudaMemcpy(hostOutputImageData, deviceoutput, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);


  wbImage_setData(outputImage, hostOutputImageData);

  wbSolution(args, outputImage);

  cudaFree(hist);
  cudaFree(cdf);
  cudaFree(deviceoutput);
  cudaFree(deviceInput);
  cudaFree(input_c);
  cudaFree(gray);
  

  return 0;

}

