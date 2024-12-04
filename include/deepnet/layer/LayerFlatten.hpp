/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// 1-D 차원으로 변환하는 레이어.
class LayerFlatten : public Layer {
  public:
    /// 생성자.
    LayerFlatten(const TensorGpu &x, int dim = 1);

    virtual const char *type(void) const override { return "Flatten"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;
};

} // namespace layer
} // namespace deepnet
