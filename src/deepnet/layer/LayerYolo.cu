/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <vector>
#include <math.h>

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

__global__ void kernel_yolo_forward(float *y, float *x,             //
    int batch, int height, int width, int channel,                  //
    int image_height, int image_width, int candidates, int classes, //
    float *anchors) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    auto n = batch * height * width * channel;

    while (i < n) {
        auto y_index = (i / (width * channel)) % height;
        auto x_index = (i / channel) % width;
        auto candidate_index = (i % channel) / (5 + classes);
        auto output_index = i % (5 + classes);

        if (output_index == 0) {
            // xc.
            y[i] = sigmoid(x[i] + x_index) * image_width / width;
        } else if (output_index == 1) {
            // yc.
            y[i] = sigmoid(x[i] + y_index) * image_height / height;
        } else if (output_index == 2) {
            // width.
            y[i] = exp(x[i]) * anchors[candidate_index * 2];
        } else if (output_index == 3) {
            // height.
            y[i] = exp(x[i]) * anchors[candidate_index * 2 + 1];
        } else if (output_index == 4) {
            // object confidence.
            y[i] = sigmoid(x[i]);
        } else {
            // class confidence.
            y[i] = x[i];
        }

        i += blockDim.x * gridDim.x;
    }
}

void gpu_yolo_forward(float *y, float *x,                           //
    int batch, int height, int width, int channel,                  //
    int image_height, int image_width, int candidates, int classes, //
    float *anchors) {

    int block = (int)((batch * height * width * channel + 511) / 512);
    if (block > 65535)
        block = 65535;

        kernel_yolo_forward<<<block, 512>>> //
            (y, x, batch, height, width, channel, //
             image_height, image_width, candidates, classes, //
             anchors);
}
