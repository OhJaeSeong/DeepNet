/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Loss.hpp"

namespace deepnet {

/// MSE 손실값을 계산하는 클래스.
///
/// diff = y - target
/// loss = torch.sum(diff * diff) / N
/// where N is the vector length.
/// dy = 2 * (y - target) / N
class LossMeanSquaredError : public Loss {
  public:
    /// 생성자.
    inline LossMeanSquaredError() {}

    /// 손실값을 얻는다.
    /// @param y 출력값(CPU).
    /// @param target 목표값(CPU).
    /// @param dy 델타 y(CPU).
    virtual float operator()(const TensorCpu &y, const TensorCpu &target, TensorCpu &dy);
};

} // namespace deepnet
