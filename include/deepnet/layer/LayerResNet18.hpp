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

/// ResNet-18을 구성하는 기본 블록.
class BasicBlock : public layer::LayerSequential {
    layer::LayerConvNorm *conv1, *conv2, *conv_down;
    layer::LayerActivationRelu *relu1, *relu2;
    layer::LayerSplit *split;
    layer::LayerMerge *merge;

  public:
    ///	생성자.
    BasicBlock(const TensorGpu &x, Workspace &workspace, //
               int channel, bool downsample = false);

    /// 소멸자.
    ~BasicBlock(){};

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "BasicBlock"; }

    /// 전방향 전파.
    void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    void backward(const TensorGpu &dy) override;
};

/// ResNet-18을 구성하는 레이어.

/// ModelResNet18에서 사용하는 특징 추출기.
class LayerResNet18 : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;

  public:
    /// 생성자.
    /// @param x 입력 텐서.
    /// @param workspace 컨볼루션 레이어에서 사용하는 작업 공간.
    LayerResNet18(const TensorGpu &x, Workspace &workspace);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "LayerResNet18"; };
};

} // namespace layer
} // namespace deepnet
