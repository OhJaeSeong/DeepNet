/// Copyright (c)2022 HanulSoft(HNS)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

class LayerCalculateEach : public Layer {

    int _type;
    /// 입력값의 포인터.
    const TensorGpu *_px2;

    /// 입력값의 차분값.
    TensorGpu _dx2;

  public:
    /// 생성자.
    LayerCalculateEach(const TensorGpu &x1, const TensorGpu &x2, int type);

    virtual const char *type(void) const override { return "CalculateEach"; }

    virtual void forward(const TensorGpu &x) override {
        DEEPNET_LOG("Unused forward");
        DEEPNET_ASSERT(false);
    }

    void forward(const TensorGpu &x1, const TensorGpu &x2);


    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter &printer, int depth = 0,
                       int index = 1) const override;
};

} // namespace layer
} // namespace deepnet
