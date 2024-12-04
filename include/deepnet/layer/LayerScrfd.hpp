/// Copyright (c)2022 HanulSoft(HNS)

#pragma once

#include <functional>

#include "deepnet/Network.hpp"
#include "deepnet/Workspace.hpp"
#include "deepnet/layer/LayerActivationRelu.hpp"
#include "deepnet/layer/LayerBatchNorm.hpp"
#include "deepnet/layer/LayerConvolutional.hpp"
#include "deepnet/layer/LayerMerge.hpp"
#include "deepnet/layer/LayerPooling.hpp"
#include "deepnet/layer/LayerSequential.hpp"
#include "deepnet/layer/LayerSplit.hpp"
#include "deepnet/layer/LayerUpSample.hpp"

namespace deepnet {
namespace layer {

class NormalSeq : public LayerSequential {
    Workspace *_workspace;
    LayerConvolutional *conv;
    LayerBatchNorm *batch;
    LayerActivationRelu *act;
public:
    NormalSeq(const TensorGpu &x, Workspace &workspace, int input_filter, int kernal, int stride, int pad, int group);

    virtual const char *type(void) const override { return "NormalSeq"; };

    virtual void forward(const TensorGpu &x) override;
};


class NormalSeqNoact : public LayerSequential {
    Workspace *_workspace;
    LayerConvolutional *conv;
    LayerBatchNorm *batch;
public:
    NormalSeqNoact(const TensorGpu &x, Workspace &workspace, int input_filter, int kernal, int stride, int pad, int group);

    virtual const char *type(void) const override { return "NormalSeqNoact"; };

    virtual void forward(const TensorGpu &x) override;
};

/// Scrfd를 구성하는 레이어1.
class Circle : public LayerSequential {
    Workspace *_workspace;
    LayerSplit *split;
    NormalSeq *conv1, *conv2;
    NormalSeqNoact *conv3;
    LayerMerge *merge;
    LayerActivationRelu *relu;

  public:
    /// 생성자.
    Circle(const TensorGpu &x, Workspace &workspace, //
                    int filter1, int filter2);

    virtual const char *type(void) const override { return "Circle"; };

    virtual void forward(const TensorGpu &x) override;
};


/// Scrfd를 구성하는 레이어2.
class ExpCircle : public LayerSequential {
    Workspace *_workspace;
    LayerSplit *split;
    NormalSeq *conv1, *conv2;
    NormalSeqNoact *conv3, *conv4; 
    LayerActivationRelu *relu;
    LayerPooling *avgPool;
    LayerMerge *merge;

  public:
    ExpCircle(const TensorGpu &x, Workspace &workspace, //
                    int filter1, int filter2, int str);

    virtual const char *type(void) const override { return "Branch"; };

    virtual void forward(const TensorGpu &x) override;
};


/// Scrfd 모델의 classify 역할을 하는 레이어.
class LayerScrfd : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;

  public:
    /// 생성자.
    /// @param x 입력 텐서.
    /// @param workspace 컨볼루션 레이어에서 사용하는 작업 공간.
    LayerScrfd(const TensorGpu &x, Workspace &workspace, std::function<void(int)> callback = nullptr);

    virtual const char *type(void) const override { return "Scrfd"; };

    NormalSeq *conv_0, *conv_2, *conv_4;
    ExpCircle *ExpCircle1, *ExpCircle2, *ExpCircle3, *ExpCircle4;
    Circle *Circle1, *Circle2, *Circle3, *Circle4, *Circle5, *Circle6, *Circle7, *Circle8, *Circle9, *Circle10, *Circle11,
        *Circle12, *Circle13, *Circle14, *Circle15, *Circle16, *Circle17, *Circle18, *Circle19, *Circle20, *Circle21,
        *Circle22, *Circle23, *Circle24, *Circle25, *Circle26, *Circle27, *Circle28, *Circle29, *Circle30, *Circle31,
        *Circle32, *Circle33, *Circle34, *Circle35, *Circle36, *Circle37, *Circle38, *Circle39;
    LayerPooling *maxpool_6;
    
    LayerMerge *merge3and2, *merge2and1, *add_150, *add_152;
    LayerUpSample *upsample3to2, *upsample2to1;

    virtual void forward(const TensorGpu &x) override;

    const TensorGpu& out1() const { return Circle31->y(); }
    
    const TensorGpu& out2() const { return Circle32->y(); }
    
    const TensorGpu& out3() const { return Circle39->y(); }
};

} // namespace layer
} // namespace deepnet
