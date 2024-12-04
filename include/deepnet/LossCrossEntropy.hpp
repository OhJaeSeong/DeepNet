/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Loss.hpp"

namespace deepnet {

/// Cross Entropy 손실값을 계산하는 클래스.
///
/// y1 = exp(y) / sum(exp(y))
/// loss = sum(-target * log(y1)) or average(-target * log(y1))
class LossCrossEntropy : public Loss {
    bool _mean;
    float _epsilon;

  public:
    /// 생성자.
    /// @param mean true이면 평균 값, false이면 합계.
    /// @param epsilon y의 입력값이 0인 경우에 오버플로우가 발생하므로
    /// 이를 막기위해 사용하는 대체 값.
    inline LossCrossEntropy(bool mean = true, float epsilon = 1E-44f) //
        : _mean(mean), _epsilon(epsilon) {
        DEEPNET_ASSERT(epsilon > 0.0f);
    }

    /// 손실값을 얻는다.
    /// @param y 출력값(CPU).
    /// @param target 목표값(CPU, one-hot 벡터).
    /// @param dy 델타 y(CPU).
    virtual float operator()(const TensorCpu &y, const TensorCpu &target,
                             TensorCpu &dy);
};

} // namespace deepnet
