// MP5 Reduction
// Input: A num list of length n
// Output: Sum of the list = list[0] + list[1] + ... + list[n-1];

#include <wb.h>

#define BLOCK_SIZE 512 //@@ This value is not fixed and you can adjust it according to the situation

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
  
__global__ void total(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  //@@ Traverse the reduction tree
  //@@ Write the computed sum of the block to the output vector at the correct index
  __shared__ float partialSum[2*BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  if(2*blockIdx.x*blockDim.x+t < len){
  partialSum[t] = input[2*blockIdx.x*blockDim.x+t];
  }
  else{
  partialSum[t] = 0;
  }
  if(2*blockIdx.x*blockDim.x+blockDim.x<len){
   partialSum[t+blockDim.x] = input[2*blockIdx.x*blockDim.x+blockDim.x+t];
  }else{
  partialSum[t+blockDim.x] = 0;
  }

  for(unsigned int stride = 1; stride <=blockDim.x; stride*=2){
    __syncthreads();
    if(t%stride == 0){
      partialSum[2*t]+= partialSum[2*t+stride];
    }

  }

  
  output[blockIdx.x] = partialSum[2*t];
 


}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  //@@ Initialize device input and output pointers

  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list
  
  args = wbArg_read(argc, argv);

  //Import data and create memory on host
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  // The number of input elements in the input is numInputElements
  // The number of output elements in the input is numOutputElements

  //@@ Allocate GPU memory
  float * input;
  float * output;
 

  cudaMalloc((void**)&input, int(sizeof(float))*numInputElements);
  cudaMalloc((void**)&output, int(sizeof(float))*numOutputElements);

  cudaMemcpy(input, hostInput, sizeof(float)*numInputElements, cudaMemcpyHostToDevice);

 dim3 dimGrid(ceil((1.0*numInputElements)/(BLOCK_SIZE)),1,1);
 dim3 dimBlock(BLOCK_SIZE,1,1);


 total<<<dimGrid, dimBlock>>> (input, output, numInputElements);
  
  cudaDeviceSynchronize();  
  //@@ Copy the GPU output memory back to the CPU
   cudaMemcpy(hostOutput,output,numOutputElements * sizeof(float) , cudaMemcpyDeviceToHost);

  
  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. 
   * For simplicity, we do not require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  //@@ Free the GPU memory
  cudaFree(input);
  cudaFree(output);



  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}

