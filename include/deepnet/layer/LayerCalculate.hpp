/// Copyright (c)2022 HanulSoft(HNS)

#pragma once

#include "deepnet/Workspace.hpp"
#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// 사칙연산(덧셈, 곱셈)을 수행하는 레이어.
class LayerCalculate : public Layer {
    /// 작업 공간의 포인터.
    Workspace *_workspace;
    int _type;
    double _number;

  public:
    /// 생성자.
    LayerCalculate(const TensorGpu &x, Workspace &workspace, double number, int type = 1);

    virtual const char *type(void) const override { return "Calculate"; }

    virtual void forward(const TensorGpu &x) override;

    virtual void backward(const TensorGpu &dy) override;

    virtual void print(tool::TablePrinter &printer, int depth = 0,
                       int index = 1) const override;
};

} // namespace layer
} // namespace deepnet
