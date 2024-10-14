#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"



#define TILE_WIDTH 16

// __global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     /*
//     Modify this function to implement the forward pass described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch
//     The goal here is to be correct AND fast.

//     Function paramter definitions:
//     output - output
//     input - input
//     mask - convolution kernel
//     Batch - batch_size (number of images in x)
//     Map_out - number of output feature maps
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */

//    __shared__ float N_ds[TILE_WIDTH+K-1][TILE_WIDTH+K-1];
   
//     const int Height_out = Height - K + 1;
//     const int Width_out = Width - K + 1;
//      int w_grid = ceil((Width_out/(1.0*TILE_WIDTH)));
//     // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
//     // An example use of these macros:
//     // float a = in_4d(0,0,0,0)
//     // out_4d(0,0,0,0) = a
//     #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
//     #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
//     #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

//     // Insert your GPU convolution kernel code here
//     int n = blockIdx.x;
//     int m = blockIdx.y;
//     int h = (blockIdx.z/w_grid)*TILE_WIDTH +threadIdx.y;

//     int w = (blockIdx.z %(w_grid))*TILE_WIDTH + threadIdx.x;

//     float acc = 0.0f;
//     if(w<Width_out && h<Height_out){
//     for(int c = 0; c<Channel; c++){
//         N_ds[p][q] = in_4d(n, c, h, w);
//         for(int p = 0; p<K; p++){
//             for(int q = 0; q<K; q++){
//                 acc += N_ds[p][q] * mask_4d(m, c, p, q);
//             }
//         }
//     }
//     }


//    if(w<Width_out && h<Height_out){
//     out_4d(n,m,h,w)= acc;
//    }
    

//     #undef out_4d
//     #undef in_4d
//     #undef mask_4d
// }



__global__ void unroll(const float *device_input, float *device_input_unroll, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K){

    int c,s,h_out, w_out, h_unroll, w_base, p, q;

    int t = blockIdx.x*1024+threadIdx.x;

   #define in_4d(i3, i2, i1, i0) device_input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    int width_out = Width-K+1;
    int height_out = Height-K+1;
    int w_unroll = height_out*width_out;

    if(t<Channel*w_unroll){
        c=t/w_unroll;
        s = t%w_unroll;
        h_out = s/width_out;
        w_out = s%width_out;
        h_unroll = h_out*width_out+w_out;
        w_base = c*K*K;
        for(p=0; p<K; p++){
            for(q=0; q<K; q++){
                int W_unroll = w_base+p*K+q;
                device_input_unroll[W_unroll*w_unroll +h_unroll] = in_4d(Batch,c,h_out+p, w_out+q);
            }
        }
    }
}
__global__ void matrixMultiplyShared(const float *A, float *B, float *C,
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

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }


    cudaMalloc((void**)device_output_ptr, (Height-K+1)*(Width-K+1)*Map_out*Batch*sizeof(float));
    cudaMalloc((void**)device_input_ptr, (Height)*(Width)*Channel*Batch*sizeof(float));
    cudaMalloc((void**)device_mask_ptr, K*K*Channel*Map_out*sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Width * Height* Channel*Batch*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, K* K* Channel*Map_out*sizeof(float), cudaMemcpyHostToDevice);


}


__host__ void GPUInterface::conv_forward_gpu( float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel




    float *device_input_unroll;
    

    int i = Batch;

    int width_out = Width-K+1;
    int height_out = Height-K+1;

    cudaMalloc((void**)&device_input_unroll, (height_out)*(width_out)*K*K*Channel*sizeof(float));

   

    // int W_grid = ceil(width_out/(1.0*TILE_WIDTH));
    // int H_grid = ceil(height_out/(1.0*TILE_WIDTH));
    int h_unroll = width_out*height_out;
    int w_unroll = Channel*K*K;




    dim3 dimGrid( ceil((1.0*width_out*height_out)/TILE_WIDTH),ceil((1.0*Map_out)/TILE_WIDTH),1);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);

    int num_blocks = ceil((1.0*Channel*width_out*height_out)/1024);
    
  
   float *output = device_output;
   for( i = 0; i<Batch; i++){
    // device_output += (Height-K+1)*(Width-K+1)*Map_out;


    // const float *device_input, float *device_input_unroll, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K
    unroll<<<num_blocks, 1024>>>(device_input, device_input_unroll, device_mask, i, Map_out, Channel, Height, Width, K);
    // cudaDeviceSynchronize();
    
    matrixMultiplyShared<<<dimGrid, dimBlock>>>(device_mask, device_input_unroll, device_output + i * Map_out*height_out*width_out,  Map_out ,Channel*K*K,w_unroll  ,h_unroll , Map_out,h_unroll );
    // cudaDeviceSynchronize();
   }
   cudaFree(device_input_unroll);


}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output,  (Height-K+1)*(Width-K+1)*Map_out*Batch* sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);

}
/*

my current thoughts:

-so we get x_unroll with k*k*channel*output so then we need to loop over batches and go from there and kinda keep
the same hting as the convolutional kernel except just do multiplication nmo sum



*/

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
