/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Weight.hpp"
#include "deepnet/layer/LayerActivationLeakyRelu.hpp"
#include "deepnet/layer/LayerConvNorm.hpp"

namespace deepnet {
namespace layer {

/// 컨볼루션, 배치 정규화, Leaky RELU를 함께 수행하는 레이어.
class LayerConvNormLeaky : public LayerConvNorm {
    float _slope;

  public:
    /// Leaky RELU를 수행하는 레이어.
    LayerActivationLeakyRelu *leaky;

  public:
    /// 생성자.
    LayerConvNormLeaky(const TensorGpu &x, Workspace &workspace, //
                       int out_channel, int height, int width,   //
                       int stride = 1, int padding = 0,          //
                       double epsilon = 0.0, float slope = 0.1f)
        : LayerConvNorm(x, workspace,               //
                        out_channel, height, width, //
                        stride, padding, epsilon),
          _slope(slope), //
          leaky(new LayerActivationLeakyRelu(norm->y(), slope)) {}

    /// 소멸자.
    virtual ~LayerConvNormLeaky();

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "ConvNormLeaky"; }

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

    /// 기울기 값을 반환한다.
    inline float slope(void) { return _slope; }
};

} // namespace layer
} // namespace deepnet
