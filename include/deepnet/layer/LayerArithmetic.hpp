/// Copyright (c)2022 HanulSoft(HNS)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

class LayerArithmetic : public Layer {

    float _number;
    int _type;
    /// 입력값의 포인터.
    const TensorGpu *_px2;

    /// 입력값의 차분값.
    TensorGpu _dx2;

  public:
    /// 생성자.
    LayerArithmetic(const TensorGpu &x, float number, int type);

    virtual const char *type(void) const override { return "Arithmetic"; }

    virtual void forward(const TensorGpu &x) override;

    virtual void print(tool::TablePrinter &printer, int depth = 0,
                       int index = 1) const override;
};

} // namespace layer
} // namespace deepnet
