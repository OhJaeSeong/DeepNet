/// Copyright (c)2021 Electronics and Telecommunications Research

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)

__device__ void i2dim(int i,                                         //
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

__device__ int dim2i(int batch, int channel, int height, int width, //
                     int b, int c, int h, int w) {
    return b * channel * height * width + c * height * width + h * width + w;
}

__global__ void kernel_concat_forward(float *y, float *x1, float *x2, //
                                      int batch, int channel1, int channel2,
                                      int height, int width) {
    long total_channel = channel1 + channel2;
    long n = batch * total_channel * height * width;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int b, c, h, w;

    while (i < n) {
        i2dim(i, batch, total_channel, height, width, b, c, h, w);

        if (c < channel1)
            y[i] = x1[dim2i(batch, channel1, height, width, b, c, h, w)];
        else
            y[i] = x2[dim2i(batch, channel2, height, width, //
                            b, c - channel1, h, w)];

        i += blockDim.x * gridDim.x;
    }
}

void gpu_concat_forward(float *y, float *x1, float *x2, //
                        int batch, int channel1, int channel2, int height,
                        int width) {

    long n = batch * (channel1 + channel2) * height * width;

    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;

    kernel_concat_forward<<<block, MAX_BLOCK_SIZE>>>(y, x1, x2, //
                                                     batch, channel1, channel2,
                                                     height, width);
}

__global__ void kernel_concat_backward(float *y, float *x1, float *x2, //
                                       int batch, int channel1, int channel2,
                                       int height, int width) {
    long total_channel = channel1 + channel2;
    long n = batch * total_channel * height * width;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int b, c, h, w;

    while (i < n) {
        i2dim(i, batch, total_channel, height, width, b, c, h, w);

        if (c < channel1)
            x1[dim2i(batch, channel1, height, width, b, c, h, w)] = y[i];
        else
            x2[dim2i(batch, channel2, height, width, b, c - channel1, h, w)] =
                y[i];

        i += blockDim.x * gridDim.x;
    }
}

void gpu_concat_backward(float *y, float *x1, float *x2, //
                         int batch, int channel1, int channel2, int height,
                         int width) {

    long n = batch * (channel1 + channel2) * height * width;

    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;

    kernel_concat_backward<<<block, MAX_BLOCK_SIZE>>>(y, x1, x2, //
                                                      batch, channel1, channel2,
                                                      height, width);
}
