// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


  __global__ void post_scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host

  // also do for two block
  int t = threadIdx.x;

  if(blockIdx.x != 0){

    if(2*blockIdx.x*blockDim.x+t < len){
      output[2*blockIdx.x*blockDim.x+t] +=  input[blockIdx.x -1];
    }
    else{
      output[2*blockIdx.x*blockDim.x+t] = 0;
    }
    if(2*blockIdx.x*blockDim.x+t+blockDim.x < len){
      output[2*blockIdx.x*blockDim.x+t+blockDim.x] += input[blockIdx.x -1];
    }
  }

}

__global__ void scan(float *input, float *output, float* auxilary, int len) {

  
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
      int t = threadIdx.x; 
    
  __shared__ float partialSum[2*BLOCK_SIZE];
  if(2*blockIdx.x*blockDim.x+t < len){
  partialSum[t] = input[2*blockIdx.x*blockDim.x+t];
  }
  else{
  partialSum[t] = 0;
  }
  if(2*blockIdx.x*blockDim.x+blockDim.x + t <len){
  partialSum[t+blockDim.x] = input[2*blockIdx.x*blockDim.x+blockDim.x+t];
  }else{
  partialSum[t+blockDim.x] = 0;
  }
  __syncthreads();

  int stride = 1;
  while(stride<2*BLOCK_SIZE){
      __syncthreads();
      int index = ((t+1)*stride*2)-1;
    if(index < 2*BLOCK_SIZE && (index-stride) >=0){
        partialSum[index] += partialSum[index-stride];
      }
      stride = stride*2;
    }
  stride = BLOCK_SIZE/2;
  
  while(stride>0){
    __syncthreads();
    int index = (t+1)*stride*2 -1;
    if((index+stride)<2*BLOCK_SIZE){
      partialSum[index+stride] += partialSum[index];
    }
    stride = stride/2;
  }
  __syncthreads();
  
  // write all elements in partial sum

  if(2*blockIdx.x*blockDim.x+t < len){
    output[2*blockIdx.x*blockDim.x+t] = partialSum[t];
  }
  if(2*blockIdx.x*blockDim.x+blockDim.x + t <len){
    output[2*blockIdx.x*blockDim.x+blockDim.x+t] = partialSum[t+blockDim.x];

  }
  if(threadIdx.x%(blockDim.x-1)== 0 && (threadIdx.x != 0)){
    auxilary[blockIdx.x] = partialSum[t+blockDim.x];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *auxilary;
  float *aux_sum;
  float *arbitrary;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.

  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&auxilary, 1024 * sizeof(float)));
  wbCheck(cudaMalloc((void **)&aux_sum, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&arbitrary, numElements * sizeof(float)));


  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil((1.0*numElements)/(2*BLOCK_SIZE)),1,1);
  dim3 dimBlock(BLOCK_SIZE,1,1);


  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce


  scan<<<dimGrid, dimBlock>>> (deviceInput, deviceOutput, auxilary, numElements);
  cudaDeviceSynchronize();
  scan<<<dimGrid, dimBlock>>> (auxilary, aux_sum,arbitrary, 1024);
  cudaDeviceSynchronize();
  post_scan<<<dimGrid, dimBlock>>> (aux_sum, deviceOutput,numElements);
  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));


  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(auxilary);
  cudaFree(aux_sum);
  cudaFree(arbitrary);
  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

