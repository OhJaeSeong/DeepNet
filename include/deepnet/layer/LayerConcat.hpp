/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// 채널을 3차원에서 합치는 레이어.
class LayerConcat : public Layer {
    /// 입력값의 포인터.
    const TensorGpu *_px2;

    /// 입력값의 차분값.
    TensorGpu _dx2;

  public:
    /// 생성자.
    LayerConcat(const TensorGpu &x1, const TensorGpu &x2);

    /// 소멸자.
    virtual ~LayerConcat();

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Concat"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override {
        DEEPNET_LOG("Unused forward");
        DEEPNET_ASSERT(false);
    }

    /// 전방향 전파.
    void forward(const TensorGpu &x1, const TensorGpu &x2);

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 델타 x 값을 반환한다.
    virtual TensorGpu &dx2(void);

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter& printer, int depth = 0, int index = 1) const override;
};

} // namespace layer
} // namespace deepnet
