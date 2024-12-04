/// Copyright (c)2021 Electronics and Telecommunications Research
/// Institute(ETRI)

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)

__global__ void kernel_leaky_relu(float *y, float *x, size_t n, float slope) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    while (i < n) {
        y[i] = (x[i] > 0) ? x[i] : slope * x[i];
        i += blockDim.x * gridDim.x;
    }
}

void gpu_leaky_relu(float *y, float *x, size_t n, float slope) {
    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;

    kernel_leaky_relu<<<block, MAX_BLOCK_SIZE>>>(y, x, n, slope);
}
