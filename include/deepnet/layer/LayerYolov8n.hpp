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
#include "deepnet/layer/LayerChunk.hpp"
#include "deepnet/layer/LayerView.hpp"
#include "deepnet/layer/LayerCalculateEach.hpp"

namespace deepnet {
namespace layer {

class C2f2 : public LayerSequential {
    Workspace *_workspace;
    bool add;

    LayerConvolutional *conv_first, *conv_final, *conv0, *conv1, *conv2, *conv3;
    LayerActivationSilu *act_first, *act_final, *act0, *act1, *act2, *act3;
    LayerChunk *chunk1, *chunk2;
    LayerCalculateEach *add1, *add2;
    LayerConcat *cat1, *cat2, *cat3;

public:
    C2f2(const TensorGpu &x, Workspace &workspace, int input_filter, bool add);

    virtual const char *type(void) const override { return "C2f2"; };

    virtual void forward(const TensorGpu &x) override;

    const TensorGpu& result() const { return act_final->y(); }
};

class C2f4 : public LayerSequential {
    Workspace *_workspace;

    LayerConvolutional *conv_first, *conv_final, *conv0, *conv1, *conv2, *conv3, *conv4, *conv5, *conv6, *conv7;
    LayerActivationSilu *act_first, *act_final, *act0, *act1, *act2, *act3, *act4, *act5, *act6, *act7;
    LayerChunk *chunk1, *chunk2;
    LayerCalculateEach *add1, *add2, *add3, *add4;
    LayerConcat *cat1, *cat2, *cat3, *cat4, *cat5;

public:
    C2f4(const TensorGpu &x, Workspace &workspace, int input_filter, bool add);

    virtual const char *type(void) const override { return "C2f4"; };

    virtual void forward(const TensorGpu &x) override;

    const TensorGpu& result() const { return act_final->y(); }
};

/// YOLO v8n의 base모델
class LayerYolov8n : public LayerSequential {
    Workspace *_workspace;

  public:
    /// @param x 
    /// @param workspace 
    LayerYolov8n(const TensorGpu &x, Workspace &workspace, std::function<void(int)> callback = nullptr);

    virtual const char *type(void) const override { return "Yolov8n"; };
    LayerConvolutional *conv_0, *conv_1, *conv_2, *conv_3, *conv_4, *sppf_conv_1, *sppf_conv_2, 
                        *conv_5, *conv_6;
    LayerActivationSilu *act_0, *act_1, *act_2, *act_3, *act_4, *sppf_act_1, *sppf_act_2, *act_5, *act_6;
    LayerPooling *maxpool1, *maxpool2, *maxpool3;
    LayerConcat *sppf_cat1, *sppf_cat2, *sppf_cat3, *branch_merge1, *branch_merge2, *branch_merge3, *branch_merge4;
    LayerUpSample *up1, *up2;
    C2f2 *c2f2_1, *c2f2_2, *c2f2_3, *c2f2_4, *c2f2_5, *c2f2_6;
    C2f4 *c2f4_1, *c2f4_2;

    virtual void forward(const TensorGpu &x) override;


    const TensorGpu& out1() const { return c2f2_4->result(); }
    const TensorGpu& out2() const { return c2f2_5->result(); }
    const TensorGpu& out3() const { return c2f2_6->result(); }
};

} // namespace layer
} // namespace deepnet
