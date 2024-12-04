#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)

__global__ void kernel_variance_forward(float *y, float *x, float* mean, int channel, int width, int height, int group){
    /*
    int loop = int(channel/group);
    int block = group * width * height;

    for(int lp = 0; lp < loop; lp +=1){
        double sam = 0;
        for(int v = 0; v < block; v += 1){
            sam += pow(x[lp * block + v] - mean[lp], 2);
        }
        sam = sam / double(block);
        y[lp] = sam;
    }*/

    int ad = blockDim.x * gridDim.x;
    int count = 0;
    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < channel * width * height; i += ad){
        if((i % (group * width * height)) == 0){
            count = i / (group * width * height);
            for(int v = 0; v < group * width * height; v += 1){
                y[count] = y[count] + pow((x[i + v] - mean[count]), 2);
            }
            y[count] = y[count]/(group * width * height);
        }
    }

}


void gpu_variance_forward(float *y, float *x1, float *x2, int channel, int width, int height, int group) {
    long n = channel * height * width;

    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;
    
    kernel_variance_forward<<<block, MAX_BLOCK_SIZE>>>(
    y, x1, x2, channel, width, height, group);

}