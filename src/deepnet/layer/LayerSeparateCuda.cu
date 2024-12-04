/// Copyright (c)2022 HanulSoft(HNS)

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"
#include <iostream>

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)


__device__ void sep_i2dim(int i,                                         //
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

__device__ int sep_dim2i(int batch, int channel, int height, int width, //
                     int b, int c, int h, int w) {
    return b * channel * height * width + c * height * width + h * width + w;
}

__global__ void kernel_separate_forward(float *x, float *y, int dim, int range1, int range2, 
                                        int channel, int height, int width) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    long n = channel * height * width;
    int b, c, h, w;

    while (i < n) {
        sep_i2dim(i, 1, channel, height, width, b, c, h, w);
        
        if(dim == 1){ // channel
            if((c >= range1) && (c < range2)){
                y[sep_dim2i(1, channel, height, width, b, c - range1, h, w)] = x[i]; 
            }
        }else if(dim == 2){ // height
            if((h >= range1) && (h < range2)){
                y[sep_dim2i(1, channel, height, width, b, c, h - range1, w)] = x[i]; 
            }
        }else if(dim == 3){ // width
            if((w >= range1) && (w < range2)){
                y[sep_dim2i(1, channel, height, width, b, c, h, w - range1)] = x[i]; 
            }
        }
        i += blockDim.x * gridDim.x;
    }
}

void gpu_separate_forward(float *x, float *y, int dim, int range1, int range2, int channel, int height, int width) {
    long n = channel * height * width;
    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;

    kernel_separate_forward<<<block, MAX_BLOCK_SIZE>>>(x, y, dim, range1, range2, channel, height, width);
}

