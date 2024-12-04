/// Copyright (c)2022 HanulSoft(HNS)

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)

__device__ void cat_i2dim(int i,                                         //
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

__device__ int cat_dim2i(int batch, int channel, int height, int width, //
                     int b, int c, int h, int w) {
    return b * channel * height * width + c * height * width + h * width + w;
}

__global__ void kernel_cat_forward(float *y, float *x1, float *x2, //
                                    int batch, int channel,
                                    int height1, int height2, int width) {
    long total_height = height1 + height2;
    long n = batch * channel * total_height * width;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int b, c, h, w;

    while (i < n) {
        cat_i2dim(i, batch, channel, total_height, width, b, c, h, w);

        if (h < height1)
            y[i] = x1[cat_dim2i(batch, channel, height1, width, b, c, h, w)];
        else
            y[i] = x2[cat_dim2i(batch, channel, height2, width, //
                            b, c, h - height1, w)];

        i += blockDim.x * gridDim.x;
    }
}

void gpu_cat_forward(float *y, float *x1, float *x2, //
                        int batch, int channel, int height1, int height2,
                        int width) {

    long n = batch * channel * (height1 + height2) * width;

    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;

    kernel_cat_forward<<<block, MAX_BLOCK_SIZE>>>(y, x1, x2, //
                                                batch, channel, 
                                                height1, height2, width);
}


__global__ void kernel_cat_backward(float *y, float *x1, float *x2, //
                                       int batch, int channel, 
                                       int height1, int height2, int width) {
    long total_height = height1 + height2;
    long n = batch * channel * total_height * width;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int b, c, h, w;

    while (i < n) {
        cat_i2dim(i, batch, channel, total_height, width, b, c, h, w);

        if (h < height1)
            x1[cat_dim2i(batch, channel, height1, width, b, c, h, w)] = y[i];
        else
            x2[cat_dim2i(batch, channel, height2, width, b, c, h - height1, w)] =
                y[i];

        i += blockDim.x * gridDim.x;
    }
}

void gpu_cat_backward(float *y, float *x1, float *x2, //
                         int batch, int channel, 
                         int height1, int height2, int width) {

    long n = batch * channel * (height1 + height2) * width;

    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;

    kernel_cat_backward<<<block, MAX_BLOCK_SIZE>>>(y, x1, x2, //
                                                      batch, channel, 
                                                      height1, height2, width);
}
