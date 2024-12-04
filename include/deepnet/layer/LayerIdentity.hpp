/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// Identity 레이어.

/// Layer의 _y(=*_px)와 _dx(=*p_dy)를 사용하지 않는다.
class LayerIdentity : public Layer {
  protected:
    /// 출력단의 dy의 포인터.
    const TensorGpu *_pdy;

  public:
    /// 생성자.
    LayerIdentity(const TensorGpu &x) : Layer(x), _pdy(&x) { _px = &x; }

    /// 소멸자.
    virtual ~LayerIdentity() override {}

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Identity"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override { _px = &x; }

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override { _pdy = &dy; }

    /// 출력값을 반환한다.
    virtual const TensorGpu &y(void) const override {
        return *(TensorGpu *)_px;
    }

    /// 델타 x 값을 반환한다.
    virtual const TensorGpu &dx(void) const override {
        return *(TensorGpu *)_pdy;
    }
};

} // namespace layer
} // namespace deepnet
