/// Copyright (c)2022 HanulSoft(HNS)

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)

__global__ void kernel_view_forward(float *y, float *x, int sum) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    while (i < sum) {
        y[i] = x[i];
        i += blockDim.x * gridDim.x;
    }
}

void gpu_view_forward(float *y, float *x, int sum) {

    int block = BLOCK_SIZE(sum);
    if (block > 65535)
        block = 65535;

    kernel_view_forward<<<block, MAX_BLOCK_SIZE>>>(
        y, x, sum);
}
