/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// Leaky RELU Activation 레이어.
class LayerActivationLeakyRelu : public Layer {
    float _slope;

  public:
    /// 생성자.
    LayerActivationLeakyRelu(const TensorGpu &x, float slope = 0.1f);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "LeakyRelu"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 기울기 값을 반환한다.
    inline float slope(void) { return _slope; }

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter& printer, int depth = 0, int index = 1) const override;
};

} // namespace layer
} // namespace deepnet
