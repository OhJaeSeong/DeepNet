/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// 두 레이어의 값을 더하는 레이어.

///  ResNet에서 사용한다. Layer의 _dx(=*p_dy)를 사용하지 않는다.
class LayerMerge : public Layer {
  protected:
    /// 출력단의 dy의 포인터.
    const TensorGpu *_pdy;

    /// 텐서 덧셈 연산 관련 정보를 저장한다.
    cudnnOpTensorDescriptor_t _op_desc;

  public:
    /// 생성자.
    LayerMerge(const TensorGpu &x);

    /// 소멸자.
    virtual ~LayerMerge();

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Merge"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override { DEEPNET_ASSERT(false); }

    /// 전방향 전파.
    void forward(const TensorGpu &x1, const TensorGpu &x2);

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 델타 x 값을 반환한다.
    virtual const TensorGpu &dx(void) const override {
        return *(TensorGpu *)_pdy;
    }
};

} // namespace layer
} // namespace deepnet
