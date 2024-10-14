#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 3  //? not sure just basing on that kernel thing which I think is the filter? 
#define MASK_WIDTH 3

//@@ Define constant memory for device kernel here

//int d = (by*gridDim.x*blockDim.x*blockDim.y+ty*blockDim.x*gridDim.x +tx) +bz*gridDim.y*gridDim.x*blockDim.x*blockDim.y*blockDim.z +tz*blockDim.x*gridDim.x*gridDim.y*gridDim.x*blockDim.x*blockDim.y;   // prob wrong but was
__constant__ float kernel[3][3][3];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  // so I am assuming this convolution on itself


  int bx = blockIdx.x; 
  int by = blockIdx.y;
  int bz = blockIdx.z;

  int tx = threadIdx.x; 
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // find in terms of 1d array out x and y anz companent

  // int x_start = bx*blockDim.x;   // gives top left 
  // int y_start = by*blockDim.y;
  // int z_start = bz*blockDim.z;

 // moves to next block in x direction 

  int x = blockDim.x*bx+tx;   // gives thread in middle
  int y = blockDim.y*by+ty;
  int z = blockDim.z*bz+tz;


  float Pvalue = 0;   // initialize output

  int radius =((MASK_WIDTH)/2); // floored division for odd mask_width


  __shared__ float N_ds[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];

  // I am going to approach with strategy 3
  // calculated exact location 

 
  // so now we have start point calculated with long equation so now we can use this to go minus two etc
  int d = z*x_size*y_size + y*x_size + x;

  // int This_tile_start_point = z_start*x_size*y_size + y_start*x_size + x_start;

  // technically just need to compare to x since either two are greater

    /// just like one cube movement to upper left hand corner? 

  int in_x = tx-radius;   // this is going to give start point
  int in_y = ty-radius;
  int in_z = tz-radius;  

  // set start of tile

// just one tile right we aint going through a bunch comprendo
  if ((y >= 0) && (y < y_size) && (x >= 0) && (x < x_size) && (z>=0)&&(z<z_size)) {
    N_ds[tz][ty][tx] = input[d];
  } 
  else {
    N_ds[tz][ty][tx] = 0.0;  // basically outside bounds
  }
  __syncthreads();

  // go through rows and columns

  for(int i = 0; i<MASK_WIDTH; i++){ // looping through dig
      
    for(int j=0; j<MASK_WIDTH; j++){//looping through rows

      for(int k = 0; k<MASK_WIDTH; k++){ //looping through columns

        int N_index_x = (in_x + k) + bx*blockDim.x;   //this seems to easy to be correct
        int N_index_y = in_y + j + by*blockDim.y;
        int N_index_z = in_z + i + bz*blockDim.z;
        int N_index = N_index_z*x_size*y_size + N_index_y*x_size + N_index_x; // index for 1-D array

        if (N_index_x >= 0 && N_index_y>=0 && N_index_z>=0 && N_index_x < x_size && N_index_y<y_size && N_index_z<z_size) {
          if ((N_index_x >= bx*blockDim.x) && (N_index_x < (bx+1)*blockDim.x) && (N_index_y >= by*blockDim.y) && (N_index_y < (by+1)*blockDim.y) && (N_index_z >= bz*blockDim.z) && (N_index_z < (bz+1)*blockDim.z)) // checking if within tile
            Pvalue += 1.0*N_ds[in_z + i][in_y + j][in_x + k] * kernel[i][j][k];
          else   //means were outside tile width but inside bounds
            Pvalue += 1.0*input[N_index] * kernel[i][j][k]; // for -2 area if not inside tile halo value so gotta do global memory
        }
        // else dont do anything as dealing with ghost
      }
    }
  }

  __syncthreads();

  // just update
  if ((y >= 0) && (y < y_size) && (x >= 0) && (x < x_size) && (z>=0)&&(z<z_size)) {
     output[d] = Pvalue;
  }


}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27); 


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  float *input;
  float *output;

  int input_size = (inputLength -3) * sizeof(float);   // maybe has to do with this
  int output_size = (inputLength-3)*sizeof(float);
  int kernel_size = kernelLength *sizeof(float);

  cudaMalloc((void **) &input, input_size);   // allocating space for the device pointers as of rn they are arbitrary
 
  cudaMalloc((void **) &output, output_size);
  
  

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu

  cudaMemcpy(input, hostInput+3, input_size, cudaMemcpyHostToDevice); // +3*sizeof(float) to skip those first rhough mem addresses
  cudaMemcpyToSymbol(kernel, hostKernel, kernel_size, 0, cudaMemcpyHostToDevice);

  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil((1.0*x_size)/TILE_WIDTH), ceil((1.0*y_size)/TILE_WIDTH), ceil((1.0*z_size)/TILE_WIDTH) ); // straight from slides 
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(input, output, z_size, y_size,x_size);

  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3,output, output_size, cudaMemcpyDeviceToHost); // +3 to set output dimensions in next step;
  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  
  wbSolution(args, hostOutput, inputLength);
  //@@ Free device memory
  cudaFree(input);
  cudaFree(output);
  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

