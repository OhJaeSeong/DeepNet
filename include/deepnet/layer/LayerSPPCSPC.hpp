/// Copyright (c)2022 HanulSoft(HNS)
#pragma once
#include <functional>

#include "deepnet/Network.hpp"
#include "deepnet/Workspace.hpp"
#include "deepnet/layer/LayerMerge.hpp"
#include "deepnet/layer/LayerPooling.hpp"
#include "deepnet/layer/LayerSequential.hpp"
#include "deepnet/layer/LayerSplit.hpp"
#include "deepnet/layer/LayerConcat.hpp"
#include "deepnet/layer/LayerUpSample.hpp"
#include "deepnet/layer/LayerActivationSilu.hpp"
#include "deepnet/layer/LayerConvolutional.hpp"
#include "deepnet/layer/LayerBatchNorm.hpp"
#include "deepnet/layer/LayerCat.hpp"
#include "deepnet/layer/LayerPermute.hpp"
#include "deepnet/layer/LayerImplicit.hpp"

namespace deepnet {
namespace layer {

class CreatedSeq : public LayerSequential {
    Workspace *_workspace;

    LayerConvolutional *conv;
    LayerBatchNorm *batch;
    LayerActivationSilu *act;
public:
    CreatedSeq(const TensorGpu &x, Workspace &workspace, int input_filter, int kernal, int stride, int pad, int group);

    virtual const char *type(void) const override { return "CreatedSeq"; };

    virtual void forward(const TensorGpu &x) override;
};


class CSPALayer : public LayerSequential {
    Workspace *_workspace;
    LayerConvolutional *conv1, *conv2, *bottle1, *bottle2, *bottle3, *bottle4, *conv3;
    LayerBatchNorm *batch1, *batch2, *bb1, *bb2, *bb3, *bb4, *batch3;
    LayerActivationSilu *act1, *act2, *ba1, *ba2, *ba3, *ba4, *act3;
    LayerConcat *cat1, *cat2, *cat3;

  public:
    CSPALayer(const TensorGpu &x, Workspace &workspace, int input_filter);

    virtual const char *type(void) const override { return "CSPALayer"; };

    virtual void forward(const TensorGpu &x) override;

    const TensorGpu& result() const { return act3->y(); }
};


class CSPBLayer : public LayerSequential {
    Workspace *_workspace;
    LayerConvolutional *conv1, *conv2, *bottle1, *bottle2, *bottle3, *bottle4, *conv3;
    LayerBatchNorm *batch1, *batch2, *bb1, *bb2, *bb3, *bb4, *batch3;
    LayerActivationSilu *act1, *act2, *ba1, *ba2, *ba3, *ba4, *act3;
    LayerConcat *cat1, *cat2, *cat3, *cat4, *cat5;

  public:
    CSPBLayer(const TensorGpu &x, Workspace &workspace, int input_filter);

    virtual const char *type(void) const override { return "CSPBLayer"; };

    virtual void forward(const TensorGpu &x) override;

    const TensorGpu& result() const { return act3->y(); }
};

class LayerCircle : public LayerSequential {
    Workspace *_workspace;
    CreatedSeq *conv1, *conv2, *conv3, *conv4, *conv5, *conv6, *conv7, *conv8, *conv9, *conv10, *conv11;
    LayerConvolutional *conv12;

  public:
    LayerCircle(const TensorGpu &x, Workspace &workspace, int input_filter);
    
    virtual const char *type(void) const override { return "LayerCircle"; };

    virtual void forward(const TensorGpu &x) override;
};

/// YOLO v7의 classify모델
class LayerSPPCSPC : public LayerSequential {
    Workspace *_workspace;

  public:
    /// @param x 
    /// @param workspace 
    LayerSPPCSPC(const TensorGpu &x, Workspace &workspace, std::function<void(int)> callback = nullptr);

    virtual const char *type(void) const override { return "SPPCSPC"; };
    CreatedSeq *conv_41, *conv_45, *conv_78, *conv_111, *conv_144, *conv_177, *conv_210, *conv_234, *conv_214, *conv_218,
                *conv_226, *conv_230, *conv_239, *conv_243, *conv_249, *conv_283, *conv_289, *conv_323, *conv_329,
                *conv_363, *conv_397, *conv_431, *conv_465, *conv_469, *conv_473, *conv_477;
    CSPALayer *CSPALayer1, *CSPALayer2, *CSPALayer3, *CSPALayer4, *CSPALayer5;
    CSPBLayer *CSPBLayer1, *CSPBLayer2, *CSPBLayer3, *CSPBLayer4, *CSPBLayer5, *CSPBLayer6;
    
    LayerConvolutional *c0, *c1, *c2, *c3;
    LayerCircle *circle1, *circle2, *circle3, *circle4;
    LayerImplicit *a0, *a1, *a2, *a3, *m0, *m1, *m2, *m3;
    LayerConcat *concat1_1, *concat1_2, *concat1_3, *concat2_1, *concat3_1, *concat3_2, *concat3_3,
                *concat4_1, *concat4_2, *concat4_3, *x6_concat, *x7_concat, *x8_concat, *x9_concat;
    LayerPooling *maxpool_222, *maxpool_223, *maxpool_224;
    LayerUpSample *up_248, *up_288, *up_328;

    virtual void forward(const TensorGpu &x) override;

    const TensorGpu& out1() const { return x6_concat->y(); }
    
    const TensorGpu& out2() const { return x7_concat->y(); }
    
    const TensorGpu& out3() const { return x8_concat->y(); }

    const TensorGpu& out4() const { return x9_concat->y(); }

};

} // namespace layer
} // namespace deepnet
