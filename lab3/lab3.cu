#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {

            
  //@@ Insert code to implement matrix multiplication here
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH]; 
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x; 
  int by = blockIdx.y;
  int tx = threadIdx.x; 
  int ty = threadIdx.y;
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;



// check both row and col
  for (int q = 0; q < ceil((1.0*numBRows)/TILE_WIDTH); ++q) {
  // Collaborative loading of M and N tiles into shared memory
    if (Row < numARows && q*TILE_WIDTH+tx < numAColumns) {
      subTileA[ty][tx] = A[Row*numAColumns + q*TILE_WIDTH+tx];
    }
     else{
    subTileA[ty][tx] = 0;
     }
    if (q*TILE_WIDTH+ty < numBRows && Col < numBColumns ){
      subTileB[ty][tx] = B[(q*TILE_WIDTH+ty)*numBColumns+Col];
    }
    else{
      subTileB[ty][tx] = 0;
    }
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k){   
        Pvalue += subTileA[ty][k] * subTileB[k][tx];
    } 
  __syncthreads();
  }


   if (Row < numCRows && Col < numCColumns) {
     C[Row*numCColumns+Col] = Pvalue;
   }

  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
 
 


  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
   numCRows = numARows;
   numCColumns = numBColumns; 
 

  //@@ Allocate the hostC matrix
  hostC =  (float *)malloc(numCColumns * numCRows* sizeof(float));

  //@@ Allocate GPU memory here
  float *deviceA;
  float *deviceB;
  float *deviceC;

  int sizeA = numARows * numAColumns * sizeof(float);   // maybe has to do with this
  int sizeB = numBRows * numBColumns * sizeof(float);
  int sizeC = numCRows * numCColumns * sizeof(float);

  cudaMalloc((void **) &deviceA, sizeA);   // allocating space for the device pointers as of rn they are arbitrary
 
  cudaMalloc((void **) &deviceB, sizeB);
  
  cudaMalloc((void **) &deviceC, sizeC);

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil((1.0*numCColumns)/TILE_WIDTH), ceil((1.0*numCRows)/TILE_WIDTH), 1); // straight from slides 
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbSolution(args, hostC, numCRows, numCColumns);
  free(hostA);
  free(hostB);
  //@@ Free the hostC matrix
  free(hostC);
  return 0;
}
