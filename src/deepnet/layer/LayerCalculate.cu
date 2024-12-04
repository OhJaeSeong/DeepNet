/// Copyright (c)2022 HanulSoft(HNS)
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)

__global__ void kernel_calculate_add(float *y, float *x,  double number, long n){
    long i = threadIdx.x + blockIdx.x * blockDim.x;

    while (i < n){
        y[i] = x[i] + number;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_calculate_abs(float *y, float *x,  double number, long n){
    long i = threadIdx.x + blockIdx.x * blockDim.x;

    while (i < n){
        y[i] = x[i] - number;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_calculate_mul(float *y, float *x,  double number, long n){
    long i = threadIdx.x + blockIdx.x * blockDim.x;

    while (i < n){
        y[i] = x[i] * number;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_calculate_div(float *y, float *x,  double number, long n){
    long i = threadIdx.x + blockIdx.x * blockDim.x;

    while (i < n){
        y[i] = x[i] / number;
        i += blockDim.x * gridDim.x;
    }
}

void gpu_calculate_forward(float *y, float *x, double number, int type, long n) {
    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;
    
    if(type == 1){
        kernel_calculate_add<<<block, MAX_BLOCK_SIZE>>>(y, x, number, n);
    }else if(type == 2){
        kernel_calculate_abs<<<block, MAX_BLOCK_SIZE>>>(y, x, number, n);
    }else if(type == 3){
        kernel_calculate_mul<<<block, MAX_BLOCK_SIZE>>>(y, x, number, n);
    }else if(type == 4){
        kernel_calculate_div<<<block, MAX_BLOCK_SIZE>>>(y, x, number, n);
    }else{
        y = x;
    }
    
}