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
#include "deepnet/layer/LayerImplicit.hpp"

namespace deepnet {
namespace layer {

class EnsembleDownC : public LayerSequential {
    Workspace *_workspace;
    LayerConvolutional *conv1, *conv2, *conv3;
    LayerActivationSilu *act1, *act2, *act3;
    LayerPooling *maxpool;
    LayerConcat *cat2layer;

  public:
    EnsembleDownC(const TensorGpu &x, Workspace &workspace, int filter1, int filter2);
    
    virtual const char *type(void) const override { return "EnsembleDownC"; };

    virtual void forward(const TensorGpu &x) override;

    const TensorGpu& result() const { return cat2layer->y(); }
};


class EnsembleCircleA : public LayerSequential {
    Workspace *_workspace;
    LayerConvolutional *conv1, *conv2, *conv3, *conv4, *conv5, *conv6, *conv7, *conv8, *conv9;
    LayerActivationSilu *act1, *act2, *act3, *act4, *act5, *act6, *act7, *act8, *act9;
    LayerConcat *cat1, *cat2, *cat3, *cat4;

  public:
    EnsembleCircleA(const TensorGpu &x, Workspace &workspace, int input_filter);
    
    virtual const char *type(void) const override { return "EnsembleCircleA"; };

    virtual void forward(const TensorGpu &x) override;

    const TensorGpu& result() const { return act9->y(); }
};

class EnsembleCircleB : public LayerSequential {
    Workspace *_workspace;
    LayerConvolutional *conv1, *conv2, *conv3, *conv4, *conv5, *conv6, *conv7, *conv8, *conv9;
    LayerActivationSilu *act1, *act2, *act3, *act4, *act5, *act6, *act7, *act8, *act9;
    LayerConcat *cat1, *cat2, *cat3, *cat4, *cat5, *cat6, *cat7;

  public:
    EnsembleCircleB(const TensorGpu &x, Workspace &workspace, int input_filter);
    
    virtual const char *type(void) const override { return "EnsembleCircleB"; };

    virtual void forward(const TensorGpu &x) override;

    const TensorGpu& result() const { return act9->y(); }
};

/// YOLO v8n의 base모델
class LayerEnsemble : public LayerSequential {
    Workspace *_workspace;
    EnsembleDownC *downc1, *downc2, *downc3, *downc4, *downc5, *downc6, *downc7, *downc8;
    EnsembleCircleA *circle1a, *circle2a, *circle3a, *circle4a, *circle5a, *circle6a, 
                  *circle7a, *circle8a, *circle9a, *circle10a;
    EnsembleCircleB *circle1b, *circle2b, *circle3b, *circle4b, *circle5b, *circle6b,
                  *circle7b, *circle8b, *circle9b, *circle10b, *circle11b, *circle12b;
    LayerConvolutional *conv_first, *sp_cv1, *sp_cv2, *sp_cv3, *sp_cv4, *sp_cv5, *sp_cv6, *sp_cv7,
                      *conv410, *conv414, *conv475, *conv479, *conv540, *conv544, *conv809, *conv812, *conv815, *conv818,
                      *conv_dummpy1, *conv_dummpy2, *conv_dummpy3, *conv_dummpy4, *conv822, *conv894, *conv966, *conv1038,
                      *conv_dummpy5, *conv_dummpy6, *conv_dummpy7, *conv_dummpy8;
    LayerActivationSilu *act_first, *sp_act1, *sp_act2, *sp_act3, *sp_act4, *sp_act5, *sp_act6, *sp_act7,
                      *act410, *act414, *act475, *act479, *act540, *act544, *act809, *act812, *act815, *act818, 
                      *act_dummpy1, *act_dummpy2, *act_dummpy3, *act_dummpy4;
    LayerPooling *sp_maxpool1, *sp_maxpool2, *sp_maxpool3;
    LayerConcat *poolcat1, *poolcat2, *poolcat3, *sp_cat, *cat417, *cat482, *cat547, *cat615, *cat683, *cat751;
    LayerUpSample *up413, *up478, *up543;
    LayerCalculateEach *add1, *add2, *add3, *add4, *add5, *add6, *add7, *add8, *add9, *add10, *add11;
    LayerImplicit *dummpy_anchor;

  public:
    /// @param x 
    /// @param workspace 
    LayerEnsemble(const TensorGpu &x, Workspace &workspace, std::function<void(int)> callback = nullptr);

    virtual const char *type(void) const override { return "Ensemble"; };

    virtual void forward(const TensorGpu &x) override;

    const TensorGpu& out1() const { return conv822->y(); }
    const TensorGpu& out2() const { return conv894->y(); }
    const TensorGpu& out3() const { return conv966->y(); }
    const TensorGpu& out4() const { return conv1038->y(); }
};

} // namespace layer
} // namespace deepnet
