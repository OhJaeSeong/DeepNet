/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// 출력값의 차원을 변경하는 레이어.
class LayerShape : public Layer {
  public:
    /// 생성자.
    LayerShape(const TensorGpu &x, Dimension dims);

    /// 소멸자.
    virtual ~LayerShape() override {}

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Shape"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;
};

} // namespace layer
} // namespace deepnet
