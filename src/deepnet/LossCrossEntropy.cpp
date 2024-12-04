/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/LossCrossEntropy.hpp"
#include "deepnet/Debug.hpp"
#include <cmath>

namespace deepnet {

/// CPU 연산으로 속도가 느리다. GPU 연산으로 바꿔야 한다.
float LossCrossEntropy::operator()(const TensorCpu &y,      //
                                   const TensorCpu &target, //
                                   TensorCpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(y.dimension() == target.dimension());
    DEEPNET_ASSERT(y.dimension() == dy.dimension());

    auto size = y.size();
    auto batch = y.batch();
    auto count = size / batch;
    auto loss_sum = 0.0f;

    auto *p_y = y.data();
    auto *p_dy = dy.data();

    // 1단계: 지수값을 계산한다.
    for (auto i = 0; i < size; i++, p_y++, p_dy++) {
        *p_dy = exp(*p_y);
    }

    // 2단계: 합으로 나눈다.
    auto *p_target = target.data();
    p_dy = dy.data();

    for (auto b = 0; b < batch; b++) {
        auto sum = 0.0f;
        auto *p_dy2 = p_dy;
        auto *p_target2 = p_target;

        // 2-1단계: 배치마다 합을 계산한다.
        for (auto c = 0; c < count; c++, p_dy++, p_target++)
            sum += *p_dy;

        // 2-2단계: 합으로 나눈다.
        if (sum > 0.0f) {
            for (auto c = 0; c < count; c++, p_dy2++, p_target2++) {
                *p_dy2 /= sum;

                if (*p_target2 >= 0.999f) {
                    loss_sum += log(*p_dy2);
                    *p_dy2 -= 1.0f;
                }
                
                if (_mean)
                    *p_dy2 /= batch;
            }
        }
    }

    if (_mean)
        loss_sum /= batch;

    return -loss_sum;
}

} // namespace deepnet
