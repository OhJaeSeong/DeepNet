/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Network.hpp"
#include "deepnet/Workspace.hpp"
#include "deepnet/layer/LayerActivationRelu.hpp"
#include "deepnet/layer/LayerConvNorm.hpp"
#include "deepnet/layer/LayerMerge.hpp"
#include "deepnet/layer/LayerSequential.hpp"
#include "deepnet/layer/LayerSplit.hpp"

namespace deepnet {
namespace layer {

/// 병목 레이어. ResNet을 구성하는 단위.
class Bottleneck : public LayerSequential {
    layer::LayerConvNorm *conv1, *conv2, *conv3, *conv_down;
    layer::LayerActivationRelu *relu1, *relu2, *relu3;
    layer::LayerSplit *split;
    layer::LayerMerge *merge;

  public:
    ///	생성자.
    Bottleneck(const TensorGpu &x, Workspace &workspace, //
               int channel1, int channel2, int channel3, //
               int stride = 1,                           //
               bool downsample = false);

    /// 소멸자.
    ~Bottleneck(){};

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Bottleneck"; }

    /// 전방향 전파.
    void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    void backward(const TensorGpu &dy) override;
};

/// ResNet-50을 구성하는 레이어.

/// ModelResNet50에서 사용하는 특징 추출기.
class LayerResNet50 : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;

  public:
    /// 생성자.
    /// @param x 입력 텐서.
    /// @param workspace 컨볼루션 레이어에서 사용하는 작업 공간.
    LayerResNet50(const TensorGpu &x, Workspace &workspace);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "LayerResNet50"; };
};

} // namespace layer
} // namespace deepnet
