/// Copyright (c)2022 HanulSoft(HNS)

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)

__global__ void kernel_exponential_forward(float *y, float *x, int channel, int width, int height, int group){
    int ad = blockDim.x * gridDim.x;
    int count = 0;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < channel * width * height; i += ad) {
        if((i % (group * width * height)) == 0){
            count = i / (group * width * height);
            for (int ins = 0; ins < group * width * height; ins += 1){
                y[count] = y[count] + x[i + ins];
            }
            y[count] = y[count]/(group * width * height);
        }
    }
}
    // 


void gpu_exponential_forward(float *y, float *x, int channel, int width, int height, int group) {
    long n = channel * height * width;
    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;
    
    kernel_exponential_forward<<<block, MAX_BLOCK_SIZE>>>(
    y, x, channel, width, height, group);

}