/// Copyright (c)2022 HanulSoft(HNS)

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)


__global__ void kernel_groupnorm_forward(float *y, float *x, float *mean, float *var, float *weight, float *bias, int group, double epsilon,
                                            int batch, int channel, int height, int width){
    
    int ad = blockDim.x * gridDim.x;
    int count = 0;
    int wgt_num = 0;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < channel * width * height; i += ad) {
        count = i / (group * width * height);
        wgt_num = i / (width * height);
        y[i] = float(float(x[i] - mean[count])/sqrt(var[count] + epsilon)) * weight[wgt_num] + bias[wgt_num];
    }

    /*
    int loop = int(channel/group);
    int c = 0;
    int block = group * width * height;
    for(int lp = 0; lp < loop; lp += 1){
        for(int g = 0; g < block; g += 1){
            c = lp * group + (g / (width * height));
            y[lp * block + g] = float(float(x[lp * block + g] - mean[lp])/sqrt(var[lp] + epsilon)) * weight[c] + bias[c];
        }
    }*/
}


void gpu_groupnorm_forward(float *y, float *x, float *mean, float *var, float *weight, float *bias, int group, double epsilon,
                         int batch, int channel, int height, int width) {
    long n = batch * channel * height * width;

    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;
    
    kernel_groupnorm_forward<<<block, MAX_BLOCK_SIZE>>>(
    y, x, mean, var, weight, bias, group, epsilon, batch, channel, height, width);

}