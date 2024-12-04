/// Copyright (c)2022 HanulSoft(HNS)

#pragma once

#include <functional>

#include "deepnet/Network.hpp"
#include "deepnet/Workspace.hpp"
#include "deepnet/layer/LayerActivationSilu.hpp"
#include "deepnet/layer/LayerCat.hpp"
#include "deepnet/layer/LayerConcat.hpp"
#include "deepnet/layer/LayerConvNormSilu.hpp"
#include "deepnet/layer/LayerFlatten.hpp"
#include "deepnet/layer/LayerMerge.hpp"
#include "deepnet/layer/LayerPermute.hpp"
#include "deepnet/layer/LayerPooling.hpp"
#include "deepnet/layer/LayerSequential.hpp"
#include "deepnet/layer/LayerSplit.hpp"
#include "deepnet/layer/LayerUpSample.hpp"

namespace deepnet {
namespace layer {

class BottleneckList : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;

    LayerSplit *split1, *split2;
    LayerConvolutional *conv1, *conv2, *conv3, *conv4;
    LayerBatchNorm *batch1, *batch2, *batch3, *batch4;
    LayerActivationSilu *act1, *act2, *act3, *act4;
    LayerMerge *merge1, *merge2;
    bool addcut;

  public:
    /// 생성자.
    BottleneckList(const TensorGpu &x, Workspace &workspace, int filter, bool shortcut);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "BottleneckList"; };

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;
};

/// CSPDarknet을 구성하는 레이어
class CSPLayer : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;

    LayerSplit *start_split;
    LayerConvolutional *conv_middle1, *conv_middle2, *conv_end;
    LayerBatchNorm *batch_middle1, *batch_middle2, *batch_end;
    LayerActivationSilu *act_middle1, *act_middle2, *act_end;
    BottleneckList *bottleneck1, *bottleneck2, *bottleneck3;
    LayerConcat *cat;
    int num_type;

  public:
    /// 생성자.
    CSPLayer(const TensorGpu &x, Workspace &workspace, int start_filter, int type, bool shortcut);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "CSPLayer"; };

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    const TensorGpu& result() const { return act_end->y(); }
};

/// CSPDarkNet을 구성하는 레이어. YOLOX의 Classify모델로 들어간다.
class LayerCSPDarknet : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;

  public:
    /// 생성자.
    /// @param x 입력 텐서.
    /// @param workspace 컨볼루션 레이어에서 사용하는 작업 공간.
    LayerCSPDarknet(const TensorGpu &x, Workspace &workspace, std::function<void(int)> callback = nullptr);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "CSPDarknet"; };
    // LayerConvNormSilu ; // *focus, *dark2
    
    LayerConvolutional  *conv_41, *dark2, *dark3, *dark4, *conv_181, *conv_184, 
                        *conv_191, *conv_216, *conv_244, *conv_272, *conv_298,
                        *cls_convs1_1, *cls_convs1_2, *cls_convs2_1, *cls_convs2_2, *cls_convs3_1, *cls_convs3_2,
                        *reg_convs1_1, *reg_convs1_2, *reg_convs2_1, *reg_convs2_2, *reg_convs3_1, *reg_convs3_2,
                        *stem1, *stem2, *stem3,
                        *cls_preds1, *cls_preds2, *cls_preds3,
                        *reg_preds1, *reg_preds2, *reg_preds3,
                        *obj_preds1, *obj_preds2, *obj_preds3;
    
    LayerBatchNorm *batch_41, *dark2_batch, *dark3_batch, *dark4_batch, *batch_181, *batch_184, *batch_191,
                   *batch_216, *batch_244, *batch_272, *batch_298,
                   *cls_batch1_1, *cls_batch1_2, *cls_batch2_1, *cls_batch2_2, *cls_batch3_1, *cls_batch3_2,
                   *reg_batch1_1, *reg_batch1_2, *reg_batch2_1, *reg_batch2_2, *reg_batch3_1, *reg_batch3_2,
                   *stem1_batch, *stem2_batch, *stem3_batch;

    LayerActivationSilu *act_41, *dark2_act, *dark3_act, *dark4_act, *act_181, *act_184, *act_191,
                        *act_216, *act_244, *act_272, *act_298,
                        *cls_act1_1, *cls_act1_2, *cls_act2_1, *cls_act2_2, *cls_act3_1, *cls_act3_2,
                        *reg_act1_1, *reg_act1_2, *reg_act2_1, *reg_act2_2, *reg_act3_1, *reg_act3_2,
                        *stem1_act, *stem2_act, *stem3_act;

    CSPLayer *CSPLayer1, *CSPLayer2, *CSPLayer3, *CSPLayer4, *CSPLayer5, *CSPLayer6, *CSPLayer7, *CSPLayer8;

    LayerPooling *maxpool_187, *maxpool_188, *maxpool_189, *cls_plus, *reg_plus;
    LayerConcat *concat_190_1, *concat_190_2, *concat_190_3, *concat_212, *concat_221, *concat_249, *concat_275,
                *concat_301, *out1_cat1, *out1_cat2, *out2_cat1, *out2_cat2, *out3_cat1, *out3_cat2;
    LayerUpSample *up_x0, *up_x1;

    LayerActivationSigmoid *obj_sig1, *obj_sig2, *obj_sig3, *cls_sig1, *cls_sig2, *cls_sig3;
    // LayerFlatten *flat1, *flat2, *flat3;
    // LayerCat *output_cat1, *output_cat2;
    // LayerPermute *permuter;

    virtual void forward(const TensorGpu &x) override;

    const TensorGpu& out1() const { return out1_cat2->y(); }
    
    const TensorGpu& out2() const { return out2_cat2->y(); }
    
    const TensorGpu& out3() const { return out3_cat2->y(); }
};

} // namespace layer
} // namespace deepnet
