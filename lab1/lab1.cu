// LAB 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here

  int i;

  i = blockIdx.x * blockDim.x + threadIdx.x; // index calculation to block/dim
  // why do we just do .x? and not include .y and .z or whatever
  if(i<len){                            // if within bounds
   out[i] = in1[i] + in2[i];  
  }
 
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  int size;   // used to malloc as we need to know the size of the object. 

  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  wbLog(TRACE, "The input length is ", inputLength);

  //@@ Allocate GPU memory here

  //device pointers
  float * in1;
  float * in2;
  float * out;

  
  size = inputLength *sizeof(float);     // why do we use floats for the pointer... 
  // cuda malloc
  cudaMalloc((void **) &in1, size);   // allocating space for the device pointers as of rn they are arbitrary
  cudaMalloc((void **) &in2, size);
  cudaMalloc((void **) &out, size);

  //@@ Copy memory to the GPU here
 // cude memcpycx
  cudaMemcpy(in1, hostInput1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(in2, hostInput2, size, cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(inputLength/256), 1, 1);
  if(0!=(inputLength%256)){
    DimGrid.x ++;
  }
  dim3 DimBlock(256,1,1);

  //@@ Launch the GPU Kernel here to perform CUDA computation
  vecAdd<<<DimGrid,DimBlock>>>(in1, in2, out, inputLength);

  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, out, size, cudaMemcpyDeviceToHost);  // inputs haven't changed so no need to do nothing
  //@@ Free the GPU memory here
  cudaFree(in1);
  cudaFree(in2);
  cudaFree(out);

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
