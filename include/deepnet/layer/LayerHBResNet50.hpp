/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Network.hpp"
#include "deepnet/Workspace.hpp"
#include "deepnet/layer/LayerConvNormRelu.hpp"
#include "deepnet/layer/LayerConvNorm.hpp"
#include "deepnet/layer/LayerMerge.hpp"
#include "deepnet/layer/LayerPooling.hpp"
#include "deepnet/layer/LayerSequential.hpp"
#include "deepnet/layer/LayerSplit.hpp"
#include "deepnet/layer/LayerUpSample.hpp"
#include "deepnet/layer/LayerActivationRelu.hpp"
#include "deepnet/layer/LayerActivationLeakyRelu.hpp"

namespace deepnet {
namespace layer {

/// 한밭대 초기모델을 위해 임시적으로 만든 resnet50을 구성하는 레이어.
class BottleneckDLoop : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;

    LayerSplit *split;
    LayerConvolutional *conv1, *conv2, *conv3, *conv_down; 
    LayerBatchNorm *batch1, *batch2, *batch3, *batch_down;
    LayerActivationRelu *relu1, *relu2, *relu3;
    LayerMerge *merge;

  public:
    /// 생성자.
    BottleneckDLoop(const TensorGpu &x, Workspace &workspace, //
                    int FrontFilter, int BackFilter, bool downsample = false, int stride = 2);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "BottleneckDLoop"; };

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;
};

class BottleneckLoop : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;

    LayerSplit *split;
    LayerConvolutional *conv1, *conv2, *conv3; 
    LayerBatchNorm *batch1, *batch2, *batch3;
    LayerActivationRelu *relu1, *relu2, *relu3;
    LayerMerge *merge;

  public:
    /// 생성자.
    BottleneckLoop(const TensorGpu &x, Workspace &workspace, //
                    int FrontFilter, int BackFilter);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "BottleneckLoop"; };

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;
};


class LayerHBResNet50 : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;

  public:
    /// 생성자.
    /// @param x 입력 텐서.
    /// @param workspace 컨볼루션 레이어에서 사용하는 작업 공간.
    LayerHBResNet50(const TensorGpu &x, Workspace &workspace);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "ResNet50ForRF"; };

    LayerConvolutional *conv_0;
    LayerBatchNorm *batch_0;
    LayerActivationRelu *relu_0;
    LayerPooling *maxPool_2;
    
    BottleneckDLoop *BottleneckLoop1, *BottleneckLoop4, *BottleneckLoop8, *BottleneckLoop14;
                    
    BottleneckLoop *BottleneckLoop2, *BottleneckLoop3, *BottleneckLoop5, *BottleneckLoop6, *BottleneckLoop7,
                *BottleneckLoop9, *BottleneckLoop10, *BottleneckLoop11, *BottleneckLoop12, *BottleneckLoop13,
                *BottleneckLoop15, *BottleneckLoop16;
    

    virtual void forward(const TensorGpu &x) override;
};

} // namespace layer
} // namespace deepnet