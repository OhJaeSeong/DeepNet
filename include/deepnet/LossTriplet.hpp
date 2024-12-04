/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Tensor.hpp"

namespace deepnet {

/// Triplet 손실 값을 계산하는 클래스.
///
/// max(|a - p| - |a - n| + alpha, 0)
class LossTriplet {
    /// 경계 거리 허용 값.
    float _alpha;

  public:
    /// 생성자.
    LossTriplet(float alpha = 1.0f) : _alpha(alpha) { DEEPNET_ASSERT(alpha > 0.0f); }

    /// 소멸자.
    virtual ~LossTriplet() {}

    /// 손실값을 얻는다.
    /// @param y 출력값(CPU).
    /// @param dy 델타 y(CPU).
    /// @returns 손실값의 평균.
    ///
    /// x의 배치 크기는 3의 배수이어야 한다.
    /// 배치에 따라 anchor, positive, negative, ...의 순서를 갖는다.
    ///
    /// Loss(A, P, N) = max(|A - P|^2 - |A - N|^2 + alpha, 0)
    virtual float operator()(const TensorCpu &y, TensorCpu &dy);
};

} // namespace deepnet
