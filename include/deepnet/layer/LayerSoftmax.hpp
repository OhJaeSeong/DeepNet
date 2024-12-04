/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// SoftMax 연산 레이어.
class LayerSoftmax : public Layer {
  int _dim;

  public:
    /// 생성자.
    LayerSoftmax(const TensorGpu &x, int dim = 0);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Softmax"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;
};

} // namespace layer
} // namespace deepnet
