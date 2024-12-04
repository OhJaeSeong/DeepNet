/// Copyright (c)2022 HanulSoft(HNS)

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)

__global__ void kernel_silu(float *y, float *x, size_t n, float *sig_num) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    while (i < n) {
        y[i] = x[i] * sig_num[i];
        i += blockDim.x * gridDim.x;
    }
}

void gpu_silu(float *y, float *x, size_t n, float *sig_num) {
    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;

    kernel_silu<<<block, MAX_BLOCK_SIZE>>>(y, x, n, sig_num);
}
