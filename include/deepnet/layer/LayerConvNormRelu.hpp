/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Weight.hpp"
#include "deepnet/layer/LayerActivationRelu.hpp"
#include "deepnet/layer/LayerConvNorm.hpp"

namespace deepnet {
namespace layer {

/// 컨볼루션, 배치 정규화, ReLU를 함께 수행하는 레이어.
class LayerConvNormRelu : public LayerConvNorm {
  public:
    /// ReLU를 수행하는 레이어.
    LayerActivationRelu *relu;

  public:
    /// 생성자.
    LayerConvNormRelu(const TensorGpu &x, Workspace &workspace, //
                       int out_channel, int height, int width,   //
                       int stride = 1, int padding = 0,          //
                       double epsilon = 0.0)
        : LayerConvNorm(x, workspace,               //
                        out_channel, height, width, //
                        stride, padding, epsilon),
          relu(new LayerActivationRelu(norm->y())) {}

    /// 소멸자.
    virtual ~LayerConvNormRelu();

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "ConvNormRelu"; }

    /// 학습 상태 여부를 설정한다.
    virtual void train(void) override;

    /// 학습 상태 여부를 설정한다.
    virtual void eval(void) override;

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 출력값을 반환한다.
    virtual const TensorGpu &y(void) const override;

    /// 델타 x 값을 반환한다.
    virtual const TensorGpu &dx(void) const override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter& printer, int depth = 0, int index = 1) const override;

    /// 레이어 출력 정보를 콘솔에 출력한다.
    virtual void debug(tool::TablePrinter& printer, int depth = 0, int index = 1) override;
};

} // namespace layer
} // namespace deepnet