/// Copyright (c)2022 HanulSoft(HNS)

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"
#include <iostream>

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)


__device__ void chunk_i2dim(int i,                                         //
                      int batch, int channel, int height, int width, //
                      int &b, int &c, int &h, int &w) {
    b = i % batch;
    
    w = i % width;
    i /= width;

    h = i % height;
    i /= height;

    c = i % channel;
    i /= channel;
}

__device__ int chunk_dim2i(int batch, int channel, int height, int width, //
                     int b, int c, int h, int w) {
    return b * channel * height * width + c * height * width + h * width + w;
}

__global__ void kernel_chunk_forward(float *x, float *y, int number, int dim, int count, 
                                        int channel, int height, int width) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    long n = channel * height * width;
    int b, c, h, w;

    while (i < n) {
        chunk_i2dim(i, 1, channel, height, width, b, c, h, w);
        
        if(dim == 1){ // channel
            if((c >= count * (channel / number)) && (c < (count + 1) * (channel / number))){
                y[chunk_dim2i(1, channel, height, width, b, c - count * (channel / number), h, w)] = x[i]; 
            }
        }else if(dim == 2){ // height
            if((h >= count * (height / number)) && (h < (count + 1) * (height / number))){
                y[chunk_dim2i(1, channel, height, width, b, c, h - count * (height / number), w)] = x[i]; 
            }
        }else if(dim == 3){ // width
            if((w >= count * (width / number)) && (w < (count + 1) * (width / number))){
                y[chunk_dim2i(1, channel, height, width, b, c, h, w - count * (width / number))] = x[i]; 
            }
        }
        i += blockDim.x * gridDim.x;
    }
}

void gpu_chunk_forward(float *x, float *y, int number, int dim, int count, int channel, int height, int width) {
    long n = channel * height * width;
    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;

    kernel_chunk_forward<<<block, MAX_BLOCK_SIZE>>>(x, y, number, dim, count, channel, height, width);
}

