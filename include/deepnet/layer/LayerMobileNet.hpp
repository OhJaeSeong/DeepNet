/// Copyright (c)2022 HanulSoft(HNS)

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

/// MobileNet의 반복구조1
class AddCircle : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;

    LayerSplit *split;
    LayerConvolutional *conv1, *conv2; 
    LayerBatchNorm *batch1, *batch2;
    LayerActivationRelu *relu1, *relu2;
    LayerMerge *merge;

  public:
    /// 생성자.
    AddCircle(const TensorGpu &x, Workspace &workspace, //
                    int filter1, int filter2);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "AddCircle"; };

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;
};

/// MobileNet의 반복구조2
class AddBranch : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;

    LayerSplit *split;
    LayerConvolutional *conv1, *conv2, *conv3; 
    LayerBatchNorm *batch1, *batch2, *batch3;
    LayerActivationRelu *relu1, *relu2;
    LayerPooling *avgPool;
    LayerMerge *merge;

  public:
    /// 생성자.
    AddBranch(const TensorGpu &x, Workspace &workspace, //
                    int filter);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "AddBranch"; };

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;
};

/// MobileNet을 구성하는 레이어. Insightface의 Retinaface의 classify모델로 들어간다.
class LayerMobileNet : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;

  public:
    /// 생성자.
    /// @param x 입력 텐서.
    /// @param workspace 컨볼루션 레이어에서 사용하는 작업 공간.
    LayerMobileNet(const TensorGpu &x, Workspace &workspace);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "MobileNet"; };

    LayerConvolutional *conv_0, *conv_3, *conv_6, *conv_103, *conv_104, *conv_105, *conv_146, *conv_147, *conv_148, *conv_149, *conv_151;
    LayerBatchNorm *batch_0, *batch_3, *batch_6, *batch_103, *batch_104, *batch_105, *batch_146, *batch_147, *batch_148, *batch_149, *batch_151;
    LayerActivationRelu *relu_0, *relu_3, *relu_6;
    LayerPooling *maxPool_9;
    
    AddCircle *AddCircle_1, *AddCircle_2, *AddCircle_3, *AddCircle_4, *AddCircle_5, *AddCircle_6, 
              *AddCircle_7, *AddCircle_8, *AddCircle_9;
    AddBranch *AddBranch_1, *AddBranch_2, *AddBranch_3;
    LayerMerge *merge3and2, *merge2and1, *add_150, *add_152;
    LayerUpSample *upsample3to2, *upsample2to1;

    virtual void forward(const TensorGpu &x) override;
};

} // namespace layer
} // namespace deepnet
