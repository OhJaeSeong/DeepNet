/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#define MAX_BLOCK_SIZE 1024
#define BLOCK_SIZE(n) (int)(((n) + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE)

// TODO: 아직 구현하지 않았음.

__global__ void kernel_cross_entropy(float *y, float *x, size_t n, float slope) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    while (i < n) {
        y[i] = (x[i] > 0) ? x[i] : slope * x[i];
        i += blockDim.x * gridDim.x;
    }
}

void gpu_cross_entropy(float *y, float *x, size_t n, float slope) {
    int block = BLOCK_SIZE(n);
    if (block > 65535)
        block = 65535;

        kernel_cross_entropy<<<block, MAX_BLOCK_SIZE>>>(y, x, n, slope);
}


// float cross_entropy(const TensorCpu &y,      //
//                     const TensorCpu &target, //
//                     TensorCpu &dy) {
//     ASSERT(y.getDimension() == target.getDimension());
//     ASSERT(y.getDimension() == dy.getDimension());

//     auto size = y.size();
//     auto batch = y.batch();
//     auto count = size / batch;
//     auto loss_sum = 0.0f;

//     auto *p_y = y.data();
//     auto *p_dy = dy.data();

//     // 1단계: 지수값을 계산한다.
//     for (auto i = 0; i < size; i++, p_y++, p_dy++) {
//         *p_dy = exp(*p_y);
//     }

//     // 2단계: 합으로 나눈다.
//     auto *p_target = target.data();
//     p_dy = dy.data();

//     for (auto b = 0; b < batch; b++) {
//         auto sum = 0.0f;
//         auto *p_dy2 = p_dy;
//         auto *p_target2 = p_target;

//         // 2-1단계: 배치마다 합을 계산한다.
//         for (auto c = 0; c < count; c++, p_dy++, p_target++)
//             sum += *p_dy;

//         // 2-2단계: 합으로 나눈다.
//         if (sum > 0.0f) {
//             for (auto c = 0; c < count; c++, p_dy2++, p_target2++) {
//                 *p_dy2 /= sum;

//                 if (*p_target2 >= 0.999f) {
//                     loss_sum += log(*p_dy2);
//                     *p_dy2 -= 1.0f;
//                 }
                
//                 if (_mean)
//                     *p_dy2 /= batch;
//             }
//         }
//     }

//     if (_mean)
//         loss_sum /= batch;

//     return -loss_sum;
// }
