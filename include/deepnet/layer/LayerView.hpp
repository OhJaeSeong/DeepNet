/// Copyright (c)2022 HanulSoft(HNS)
#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

class LayerView : public Layer {
    int _order_0;
    int _order_1;
    int _order_2;
    int _order_3;

  public:
    /// 생성자.
    /// @param x
    /// @param order_0
    /// @param order_1
    /// @param order_2
    /// @param order_3
    LayerView(const TensorGpu &x, int order_0, int order_1, int order_2, int order_3);

    /// 소멸자.
    inline ~LayerView() {};

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "View"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter& printer, int depth = 0, int index = 1) const override;
};

} // namespace layer
} // namespace deepnet