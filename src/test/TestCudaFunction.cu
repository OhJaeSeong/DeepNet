/// Copyright (c)2021 Electronics and Telecommunications Research
/// Institute(ETRI)

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"

#define N 10

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)

__global__ void kernel_cuda_function(float *y, float *x, int n) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    y[i] = 100.0f;

    while (i < n) {
        y[i] = 10.0f * x[i];
        i += blockDim.x * gridDim.x;
        printf("Hello World from GPU!\n");
    }
}

void test_cuda_function(float *y, float *x, int size) {
    int block = BLOCK_SIZE(size);
    if (block > 65535)
        block = 65535;

    kernel_cuda_function<<<block, MAX_BLOCK_SIZE>>>(y, x, size);
}
