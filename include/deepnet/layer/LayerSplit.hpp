/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// ResNet의 redisual 연결을 구성할 때 사용하는 레이어.

/// 역방향 전파시 두 개의 출력으로부터 받은 dy 값을 더하는 기능을 수행한다.
/// Layer의 _y(=*_px)를 사용하지 않는다.
class LayerSplit : public Layer {
  protected:
    /// 역방향 전파시 dy1과 dy2를 더하는 연산에 관한 정보를 저장한다.
    cudnnOpTensorDescriptor_t _op_desc;

  public:
    /// 생성자.
    LayerSplit(const TensorGpu &x);

    /// 소멸자.
    virtual ~LayerSplit();

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Split"; }

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override { DEEPNET_ASSERT(false); }

    /// 역방향 전파.
    void backward(const TensorGpu &dy1, const TensorGpu &dy2);

    /// 출력값을 반환한다.
    virtual const TensorGpu &y(void) const override {
        return *(TensorGpu *)_px;
    }
};

} // namespace layer
} // namespace deepnet
