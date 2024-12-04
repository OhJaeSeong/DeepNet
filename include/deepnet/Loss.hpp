/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Tensor.hpp"

namespace deepnet {

/// 손실 값을 계산하는 클래스.
class Loss {
  public:
    /// 생성자.
    inline Loss() {}

    /// 소멸자.
    virtual ~Loss() {}

    /// 손실값을 얻는다.
    /// @param y 출력값.
    /// @param target 목표값.
    /// @param dy 델타 y.
    virtual float operator()(const TensorCpu &y, const TensorCpu &target,
                             TensorCpu &dy) {
        return 0.0f;
    }

    /// 손실값을 얻는다.
    /// @param y 출력값.
    /// @param target 목표값.
    /// @param dy 델타 y.
    virtual float operator()(const TensorGpu &y, const TensorGpu &target,
                             TensorGpu &dy) {
        return 0.0f;
    }
};

} // namespace deepnet
