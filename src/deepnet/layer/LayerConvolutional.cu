/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"
#include <iostream>

#define MAX_BLOCK_SIZE 1024
#define MAX_GRID_SIZE 65535
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)

struct ConvolutionParams {
    float *y, *x, *k;
    int batch;
    int y_height, y_width, out_channel;
    int x_height, x_width, in_channel;
    int k_height, k_width;
    int stride, padding;
};

__global__ void kernel_convolutional_forward(ConvolutionParams p) {

    int y_index = threadIdx.x + blockIdx.x * blockDim.x;
    int n = p.batch * p.y_height * p.y_width * p.out_channel;

    while (y_index < n) {
        auto sum = 0.0f;

        int b = y_index / (p.y_height * p.y_width * p.out_channel);
        int out_y = (y_index / (p.y_width * p.out_channel)) % p.y_height;
        int out_x = (y_index / p.out_channel) % p.y_width;
        int out_c = y_index % p.out_channel;

        int x_index_0 = b * p.x_height * p.x_width * p.in_channel;
        int k_index_0 = out_c * p.k_height * p.k_width * p.in_channel;

        for (auto k_y = 0; k_y < p.k_height; k_y++) {
            int yy = (k_y + out_y - p.padding) * p.stride;
            if (yy < 0 || yy >= p.k_height)
                continue;

            int x_index_1 = x_index_0 + yy * p.x_width * p.in_channel;
            int k_index_1 = k_index_0 + k_y * p.k_width * p.in_channel;

            for (auto k_x = 0; k_x < p.k_width; k_x++) {
                int xx = (k_x + out_x - p.padding) * p.stride;
                if (xx < 0 || xx >= p.k_width)
                    continue;

                int x_index_2 = x_index_1 + xx * p.in_channel;
                int k_index_2 = k_index_1 + k_x * p.in_channel;

                for (auto k_c = 0; k_c < p.in_channel; k_c++) {
                    sum += p.x[x_index_2 + k_c] * p.k[k_index_2 + k_c];
                }
            }
        }

        p.y[y_index] = sum;
        y_index += blockDim.x * gridDim.x;
    }
}

void gpu_convolutional_forward(                 //
    float *y, float *x, float *k,               //
    int batch,                                  //
    int y_height, int y_width, int out_channel, //
    int x_height, int x_width, int in_channel,  //
    int k_height, int k_width,                  //
    int stride, int padding) {

    int n = batch * y_height * y_width * out_channel;
    int block = BLOCK_SIZE(n);
    if (block > MAX_GRID_SIZE)
        block = MAX_GRID_SIZE;

    ConvolutionParams p;
    p.y = y;
    p.x = x;
    p.k = k;
    p.batch = batch;
    p.y_height = y_height;
    p.y_width = y_width;
    p.out_channel = out_channel;
    p.x_height = x_height;
    p.x_width = x_width;
    p.in_channel = in_channel;
    p.k_height = k_height;
    p.k_width = k_width;
    p.stride = stride;
    p.padding = padding;

    // TODO: NCHW
    exit(0);

    kernel_convolutional_forward<<<block, MAX_BLOCK_SIZE>>>(p);
}
