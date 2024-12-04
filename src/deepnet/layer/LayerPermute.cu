/// Copyright (c)2021 Electronics and Telecommunications Research
/// Institute(ETRI)

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

__global__ void kernel_permute_forward(float *y, float *x, //
                                       int input_0, int input_1, int input_2,
                                       int input_3, //
                                       int output_0, int output_1, int output_2,
                                       int output_3, //
                                       int order_0, int order_1, int order_2,
                                       int order_3) {

    long n = input_0 * input_1 * input_2 * input_3;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int i0, i1, i2, i3;

    while (i < n) {
        i2dim(i, input_0, input_1, input_2, input_3, i0, i1, i2, i3);

        int input_index[4] = {i0, i1, i2, i3};

        y[input_index[order_0] * output_1 * output_2 * output_3 +
          input_index[order_1] * output_2 * output_3 +
          input_index[order_2] * output_3 + input_index[order_3]] =
            x[i0 * input_1 * input_2 * input_3 + i1 * input_2 * input_3 +
              i2 * input_3 + i3];

        i += blockDim.x * gridDim.x;
    }
}

void gpu_permute_forward(float *y, float *x,                                 //
                         int input_0, int input_1, int input_2, int input_3, //
                         int output_0, int output_1, int output_2,
                         int output_3, //
                         int order_0, int order_1, int order_2, int order_3) {

    long n = input_0 * input_1 * input_2 * input_3;

    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;

    kernel_permute_forward<<<block, MAX_BLOCK_SIZE>>>(
        y, x, //
        input_0, input_1, input_2,
        input_3, //
        output_0, output_1, output_2,
        output_3, //
        order_0, order_1, order_2, order_3);
}
