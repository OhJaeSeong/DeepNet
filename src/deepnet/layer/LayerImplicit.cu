/// Copyright (c)2022 HanulSoft(HNS)

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)

static __device__ void i2dim(int i,                                         //
                             int batch, int channel, int height, int width, //
                             int &b, int &c, int &h, int &w) {
    w = i % width;
    i /= width;

    h = i % height;
    i /= height;

    c = i % channel;
    i /= channel;

    b = i % batch;
}

__global__ void kernel_implicit_forward_add(float *y, float *x, float *weight, 
                                            int batch, int channel, int height, int width){
    long n = (long)batch * channel * height * width;
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    int b, c, h, w;

    while (i < n){
        i2dim(i, batch, channel, height, width, b, c, h, w);
        y[i] = x[i] + weight[c];
        i += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_implicit_forward_mul(float *y, float *x, float *weight,  
                                            int batch, int channel, int height, int width) {
    long n = (long)batch * channel * height * width;
    long i = threadIdx.x + blockIdx.x * blockDim.x;
    int b, c, h, w;

    while (i < n){
        i2dim(i, batch, channel, height, width, b, c, h, w);
        y[i] = x[i] * weight[c];
        i += blockDim.x * gridDim.x;
    }
}

void gpu_implicit_forward(float *y, float *x, float *weight,
                         int batch, int channel, int height, int width, bool isadd) {
    long n = batch * channel * height * width;

    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;
    
    if(isadd == true){
        kernel_implicit_forward_add<<<block, MAX_BLOCK_SIZE>>>(
        y, x, weight, batch, channel, height, width);
    }else{
        kernel_implicit_forward_mul<<<block, MAX_BLOCK_SIZE>>>(
        y, x, weight, batch, channel, height, width);
    }
    
}