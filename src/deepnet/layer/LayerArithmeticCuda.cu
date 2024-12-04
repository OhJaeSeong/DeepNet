/// Copyright (c)2022 HanulSoft(HNS)

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"
#include <iostream>

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)

__global__ void kernel_arithmetic_forward(float *x, float *y, int tensor_size, float number, int type) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    long n = tensor_size;
    int b, c, h, w;

    while (i < n) {
        if(type == 0){
            y[i] = x[i] + number;
        }else if(type == 1){
            y[i] = x[i] - number;
        }else if(type == 2){
            y[i] = x[i] * number;
        }else if(type == 3){
            y[i] = x[i] / number;
        }
        i += blockDim.x * gridDim.x;
    }
}

void gpu_arithmetic_forward(float *x, float *y, int tensor_size, float number, int type) {
    long n = tensor_size;
    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;

    kernel_arithmetic_forward<<<block, MAX_BLOCK_SIZE>>>(x, y, tensor_size, number, type);
}

