/// Copyright (c)2022 HanulSoft(HNS)

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"
#include <iostream>

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)


__device__ void calce_i2dim(int i,                                         //
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

__device__ int calce_dim2i(int batch, int channel, int height, int width, //
                     int b, int c, int h, int w) {
    return b * channel * height * width + c * height * width + h * width + w;
}

__global__ void kernel_calculateE_forward(float *x1, float *x2, float *y, int channel, int height, int width, int type, int duple) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    long n = channel * height * width;
    int b, c, h, w;

    while (i < n) {
        calce_i2dim(i, 1, channel, height, width, b, c, h, w);
        if(type == 0){
            y[i] = x1[i] + x2[i];
        }else if(type == 1){
            y[i] = x1[i] - x2[i];
        }else if(type == 2){
            y[i] = x1[i] * x2[i];
        }else if(type == 3){
            y[i] = x1[i] / x2[i];
        }
        i += blockDim.x * gridDim.x;
    }
}

void gpu_calculateE_forward(float *x1, float *x2, float *y, int channel, int height, int width, int type, int duple) {
    long n = channel * height * width;
    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;

    kernel_calculateE_forward<<<block, MAX_BLOCK_SIZE>>>(x1, x2, y, channel, height, width, type, duple);
}

