/// Copyright (c)2022 HanulSoft(HNS)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

class LayerSeparate : public Layer {

    int _dim;
    int _range1;
    int _range2;

  public:
    /// 생성자.
    LayerSeparate(const TensorGpu &x, int dim, int range1, int range2);

    virtual const char *type(void) const override { return "Separate"; }

    virtual void forward(const TensorGpu &x) override;

    // void forward(const TensorGpu &x, int number, int dim, int count);


    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter &printer, int depth = 0,
                       int index = 1) const override;
};

} // namespace layer
} // namespace deepnet
