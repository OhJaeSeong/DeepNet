/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// Activation 레이어 클래스.
class LayerActivation : public Layer {
  protected:
    /// 활성화 함수의 정보를 저장한다.
    cudnnActivationDescriptor_t _act_desc;

  public:
    /// 생성자.
    LayerActivation(const TensorGpu &x);

    /// 소멸자.
    virtual ~LayerActivation();

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;
};

} // namespace layer
} // namespace deepnet
