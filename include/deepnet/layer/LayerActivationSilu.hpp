/// Copyright (c)2022 HanulSoft(HNS)
#pragma once

#include "deepnet/layer/Layer.hpp"
#include "deepnet/layer/LayerActivationSigmoid.hpp"

namespace deepnet {
namespace layer {

/// SiLU Activation 레이어.
class LayerActivationSilu : public Layer {
  
  LayerActivationSigmoid *sig;
  public:
    /// 생성자.
    LayerActivationSilu(const TensorGpu &x) 
    : Layer(x), sig(new LayerActivationSigmoid(x)) { _y.setDimension(x.dimension()); }

    virtual ~LayerActivationSilu();

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Silu"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter& printer, int depth = 0, int index = 1) const override;
};

} // namespace layer
} // namespace deepnet