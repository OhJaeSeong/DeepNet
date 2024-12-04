/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/LayerActivation.hpp"

namespace deepnet {
namespace layer {

/// ELU Activation 레이어.
class LayerActivationElu : public LayerActivation {
  public:
    /// 생성자.
    LayerActivationElu(const TensorGpu &x);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override {
        return "Elu";
    }
};

} // namespace layer
} // namespace deepnet
