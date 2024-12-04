/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {


class LayerVariance : public Layer {

  public:
    /// 생성자.
    LayerVariance(const TensorGpu &x1, const TensorGpu &x2); // x , mean(expo)

    virtual const char *type(void) const override { return "Variance"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override {
        DEEPNET_LOG("Unused forward");
        DEEPNET_ASSERT(false);
    }

    /// 전방향 전파.
    void forward(const TensorGpu &x1, const TensorGpu &x2);

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter &printer, int depth = 0,
                       int index = 1) const override;

};

} // namespace layer
} // namespace deepnet
