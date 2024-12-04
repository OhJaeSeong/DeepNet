/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"
#include <iostream>

#define MAX_BLOCK_SIZE 1024
#define MAX_GRID_SIZE 65535
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)

struct ConvTransposeParams {
    float *y, *x, *w, *b;
    int batch;
    int y_height, y_width, out_channel;
    int x_height, x_width, in_channel;
    int k_height, k_width;
    int stride, padding;
};

__global__ void kernel_convtranspose_forward(ConvTransposeParams p) {
    int y_index = threadIdx.x + blockIdx.x * blockDim.x;
    int n = p.batch * p.y_height * p.y_width * p.out_channel;

    int input_size = p.x_height * p.x_width;
    int kernal_size = p.k_height * p.k_width;
    
    int w_target = 0;
    int out_c = 0;
    int kernal_target = 0;
    
    auto sum = 0.0f;

    while (y_index < n) {
        sum = 0.0f;
        kernal_target = (y_index % p.k_width) + p.k_width * (int(y_index / p.x_width) % p.k_height); // 0 ~ kernal_size
        out_c = int(y_index / input_size) * kernal_size;

        for(int in_c = 0; in_c < p.in_channel; in_c += 1){
            sum += p.x[in_c * input_size + (y_index % input_size)] * p.w[in_c * p.out_channel * kernal_size + out_c + kernal_target];
        }
        if(p.b != NULL){
            sum += p.b[int(y_index / input_size)];
        }

        p.y[y_index] = sum;
        y_index += blockDim.x * gridDim.x;
    }
}

void gpu_convtranspose_forward(                 //
    float *y, float *x, float *w, float *b, int batch,    //
    int y_height, int y_width, int out_channel, //
    int x_height, int x_width, int in_channel,  //
    int k_height, int k_width, int stride, int padding) {

    int n = batch * y_height * y_width * out_channel;
    int block = BLOCK_SIZE(n);
    if (block > MAX_GRID_SIZE)
        block = MAX_GRID_SIZE;
    
    ConvTransposeParams p;
    p.y = y;
    p.x = x;
    p.w = w;
    p.b = b;
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

    kernel_convtranspose_forward<<<block, MAX_BLOCK_SIZE>>>(p);
}
