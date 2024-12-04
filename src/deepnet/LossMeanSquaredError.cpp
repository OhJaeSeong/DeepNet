/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/LossMeanSquaredError.hpp"
#include "deepnet/Debug.hpp"
#include <cmath>

namespace deepnet {

/// CPU 연산으로 속도가 느리다. GPU 연산으로 바꿔야 한다.
float LossMeanSquaredError::operator()(const TensorCpu &y,      //
                                       const TensorCpu &target, //
                                       TensorCpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(y.dimension() == target.dimension());
    DEEPNET_ASSERT(y.dimension() == dy.dimension());

    auto batch = y.batch();
    auto size = y.size();
    auto count = size / batch;
    auto total_sum = 0.0f;
    auto *p_y = y.data();
    auto *p_target = target.data();
    auto *p_dy = dy.data();

    auto scale = 2.0f / size;

    for (auto b = 0; b < batch; b++) {
        auto batch_sum = 0.0f;

        for (auto i = 0; i < count; i++, p_y++, p_target++, p_dy++) {
            auto diff = *p_dy = *p_y - *p_target;
            batch_sum += diff * diff;
        }

        // 길이를 계산한다.
        total_sum += batch_sum;

        // 포인터를 배치의 시작 위치로 이동한다.
        p_dy -= count;

        // -0.2667
        for (auto i = 0; i < count; i++, p_dy++)
            *p_dy *= scale;
    }

    return total_sum / size;
}

} // namespace deepnet
