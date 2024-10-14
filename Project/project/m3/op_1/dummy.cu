#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16




__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    
    extern __shared__ float N_ds[TILE_WIDTH+6][TILE_WIDTH+6];
   
   

   

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int w_grid = ceil((Width_out/(1.0*TILE_WIDTH)));
    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int n = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z/w_grid)*TILE_WIDTH +threadIdx.y;

    int w = (blockIdx.z %(w_grid))*TILE_WIDTH + threadIdx.x;

    int start_h = h;
    int start_w = w;

    float acc = 0.0f;
  
    for(int c = 0; c<Channel; c++){
        if(start_w<Width && start_h<Height  && start_w>=0 &&start_h>=0){
            N_ds[threadIdx.y][threadIdx.x] = in_4d(n, c, start_h, start_w);
        }
        else{
            N_ds[threadIdx.y][threadIdx.x] = 0;
        }
        if((6) > threadIdx.x && (6)>threadIdx.y ){
            int new_w = start_w+TILE_WIDTH;
            int new_h = start_h + TILE_WIDTH;
            if(new_w<Width && new_h<Height){
                N_ds[threadIdx.y+TILE_WIDTH][threadIdx.x+TILE_WIDTH] = in_4d(n, c, new_h, new_w);
            }
            else{
                N_ds[threadIdx.y+TILE_WIDTH][threadIdx.x+TILE_WIDTH] = 0;
            }
        }
        if ((6)>threadIdx.y){
            int new_h = start_h + TILE_WIDTH; 
            if(start_w<Width && new_h<Height){
                N_ds[threadIdx.y+TILE_WIDTH][threadIdx.x] = in_4d(n, c, new_h, start_w);
            }
            else{
                N_ds[threadIdx.y+TILE_WIDTH][threadIdx.x] = 0;
            }

        }
        if ((6) > threadIdx.x ){
            int new_w =start_w+TILE_WIDTH;
             if(new_w<Width && start_h<Height){
                N_ds[threadIdx.y][threadIdx.x+TILE_WIDTH] = in_4d(n, c, start_h, new_w);
            }
            else{
                N_ds[threadIdx.y][threadIdx.x+TILE_WIDTH] = 0;
            }

        }
        __syncthreads();
        for(int p = 0; p<K; p++){
            for(int q = 0; q<K; q++){
                acc += N_ds[p+threadIdx.y][q+threadIdx.x] * mask_4d(m, c, p, q);
            }
        }
        __syncthreads();

        
    }
    

    //3441786
    


   if(w<Width_out && h<Height_out){
    out_4d(n,m,h,w)= acc;
   }
    

    #undef out_4d
    #undef in_4d
    #undef mask_4d

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


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel

    int i = int(Batch);

    int width_out = Width-K+1;
    int height_out = Height-K+1;

    int W_grid = ceil(width_out/(1.0*TILE_WIDTH));
    int H_grid = ceil(height_out/(1.0*TILE_WIDTH));

    int Z = W_grid*H_grid;

    dim3 dimGrid(Batch,Map_out,Z);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);

    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();

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
