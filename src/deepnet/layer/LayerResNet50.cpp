/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerResNet50.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/layer/LayerConvNorm.hpp"
#include "deepnet/layer/LayerPooling.hpp"

namespace deepnet {
namespace layer {

Bottleneck::Bottleneck(const TensorGpu &x, Workspace &workspace, //
                       int channel1, int channel2, int channel3, //
                       int stride,                               //
                       bool downsample)
    : LayerSequential(x), conv_down(nullptr) {
    DEEPNET_TRACER;

    *this += split = new layer::LayerSplit(x);

    *this += conv1 = new layer::LayerConvNorm(y(), workspace, //
                                              channel1, 1, 1, 1, 0);
    *this += relu1 = new layer::LayerActivationRelu(y());

    *this += conv2 = new layer::LayerConvNorm(y(), workspace, //
                                              channel2, 3, 3, stride, 1);
    *this += relu2 = new layer::LayerActivationRelu(y());

    *this += conv3 = new layer::LayerConvNorm(y(), workspace, //
                                              channel3, 1, 1, 1, 0);

    if (downsample) {
        auto stride = (split->y().height() == conv3->y().height()) ? 1 : 2;
        *this += conv_down = new layer::LayerConvNorm( //
            split->y(), workspace, conv3->y().channel(), 1, 1, stride, 0);
    }

    *this += merge = new layer::LayerMerge(conv3->y());

    *this += relu3 = new layer::LayerActivationRelu(y());
}

void Bottleneck::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!x.isEmpty());

    x >> split            //
        >> conv1 >> relu1 //
        >> conv2 >> relu2 //
        >> conv3;

    auto *identity = &split->y();

    if (conv_down)
        identity = &((*identity) >> conv_down);

    merge->forward(conv3->y(), *identity);
    merge->y() >> relu3;
}

void Bottleneck::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!dy.isEmpty());

    dy << relu3 << merge;

    auto *identity = &merge->dx();

    if (conv_down)
        identity = &((*identity) << conv_down);

    merge->dx() << conv3          //
                << relu2 << conv2 //
                << relu1 << conv1;

    split->backward(conv1->dx(), *identity);
}

LayerResNet50::LayerResNet50(const TensorGpu &x, Workspace &workspace)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;

    try {
        *this += new layer::LayerConvNorm(x, *_workspace, 64, 7, 7, 2, 3);
        *this += new layer::LayerActivationRelu(y());
        *this += new layer::LayerPooling(y(), 3, 2, 1);

        int params[4][4] = {
            {3, 64, 64, 256},
            {4, 128, 128, 512},
            {6, 256, 256, 1024},
            {3, 512, 512, 2048},
        };

        for (auto i = 0; i < 4; i++) {
            int n = params[i][0],  //
                c1 = params[i][1], //
                c2 = params[i][2], //
                c3 = params[i][3];

            for (auto j = 0; j < n; j++) {
                auto stride = (i > 0 && j == 0) ? 2 : 1;
                *this += new Bottleneck(y(), *_workspace, //
                                        c1, c2, c3, stride, j == 0);
            }
        }
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

} // namespace layer
} // namespace deepnet
