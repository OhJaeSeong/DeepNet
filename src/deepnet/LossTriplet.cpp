/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/LossTriplet.hpp"
#include "deepnet/Debug.hpp"
#include <cmath>

namespace deepnet {

/// CPU 연산으로 속도가 느리다. GPU 연산으로 바꿔야 한다.
float LossTriplet::operator()(const TensorCpu &y, TensorCpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!y.isEmpty());
    DEEPNET_ASSERT(y.dimension() == dy.dimension());

    // 배치의 수는 3의 배수이야야 한다.
    auto batch = y.batch();
    DEEPNET_ASSERT(batch % 3 == 0);
    auto sample_size = batch / 3;
    DEEPNET_ASSERT(sample_size > 0);

    // 배치 하나의 데이터 수(= channel * height * width).
    auto data_count = y.size() / batch;

    auto p_y = y.data();
    auto p_dy = dy.data();

    auto loss_total = 0.0f;

    for (auto b = 0; b < sample_size; b++) {
        auto p_anchor = &p_y[b * 3 * data_count];
        auto p_positive = p_anchor + data_count;
        auto p_negative = p_anchor + 2 * data_count;
        auto p_da = &p_dy[b * 3 * data_count];
        auto p_dp = p_da + data_count;
        auto p_dn = p_da + 2 * data_count;

        // 1 단계: dp = p - a.
        //         dn = a - n.
        for (auto i = 0; i < data_count; i++) {
            p_dp[i] = p_positive[i] - p_anchor[i];
            p_dn[i] = p_anchor[i] - p_negative[i];
        }

        // 2 단계: loss_positive = |dp| = |p - a|,
        //         loss_negative = |dn| = |a - n|.
        //         loss_batch = max(loss_positive - loss_negative + alpha, 0)
        auto sum_positive = 0.0f;
        auto sum_negative = 0.0f;

        for (auto i = 0; i < data_count; i++) {
            sum_positive += p_dp[i] * p_dp[i];
            sum_negative += p_dn[i] * p_dn[i];
        }

        // 0으로 나누는 것을 방지한다.
        if (sum_positive == 0.0f)
            sum_positive = 0.0000001f;
        if (sum_negative == 0.0f)
            sum_negative = 0.0000001f;

        auto loss_positive = sqrt(sum_positive);
        auto loss_negative = sqrt(sum_negative);
        auto sum = loss_positive - loss_negative + _alpha;
        auto loss_batch = (sum < 0.0f) ? 0.0f : sum;

        // 3 단계: loss_total = sum(loss_batch)
        loss_total += loss_batch;

        // 4 단계: dp = dp / loss_positive,
        //         dn = dn / loss_negative,
        //         da = -dp - dn.
        for (auto i = 0; i < data_count; i++) {
            p_dp[i] /= loss_positive;
            p_dn[i] /= loss_negative;
            p_da[i] = -p_dp[i] - p_dn[i];
        }
    }

    return loss_total / sample_size;
}

} // namespace deepnet
