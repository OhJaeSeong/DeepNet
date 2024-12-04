/// Copyright (c)2022 HanulSoft(HNS)

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"
#include <iostream>

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)

__global__ void kernel_yoloanchor_forward(float *anchor, float *stride, int end1, int end2, float grid_cell_offset, int str) {
    for(int h = 0; h < end1; h +=1){
        for(int w = 0; w < end2; w +=1){
            anchor[2 * (h * end2 + w)] = w + grid_cell_offset;
        }
    }
    for(int h = 0; h < end1; h +=1){
        for(int w = 0; w < end2; w +=1){
            anchor[2 * (h * end2 + w) + 1] = h + grid_cell_offset;
        }
    }

    for(int n = 0; n < 4 * end1 * end2; n += 1){
        stride[n] = str;
    }
}

void gpu_yoloanchor_forward(float *anchor, float *stride, int end1, int end2, float grid_cell_offset, int str) {
    long n = end1 * end2  * 2;
    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;

    kernel_yoloanchor_forward<<<block, MAX_BLOCK_SIZE>>>(anchor, stride, end1, end2, grid_cell_offset, str);
}

