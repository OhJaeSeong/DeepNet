/// Copyright (c)2021 Electronics and Telecommunications Research
/// Institute(ETRI)

#include "deepnet/layer/LayerResNet18.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/layer/LayerConvNorm.hpp"
#include "deepnet/layer/LayerPooling.hpp"

// See
// https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
// for the ResNet example.

namespace deepnet {
namespace layer {

BasicBlock::BasicBlock(const TensorGpu &x, Workspace &workspace, int channel,
                       bool downsample)
    : LayerSequential(x), conv_down(nullptr) {
    DEEPNET_TRACER;

    *this += split = new layer::LayerSplit(x);

    auto stride = downsample ? 2 : 1;

    *this += conv1 = new layer::LayerConvNorm( //
        y(), workspace, channel, 3, 3, stride, 1);
    *this += relu1 = new layer::LayerActivationRelu(y());

    *this += conv2 = new layer::LayerConvNorm( //
        y(), workspace, channel, 3, 3, 1, 1);

    if (downsample) {
        *this += conv_down = new layer::LayerConvNorm( //
            split->y(), workspace, x.channel() * 2, 1, 1, stride);
    }

    *this += merge = new layer::LayerMerge(conv2->y());

    *this += relu2 = new layer::LayerActivationRelu(y());
}

void BasicBlock::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!x.isEmpty());

    x >> split >> conv1 >> relu1 >> conv2;

    auto *identity = &split->y();

    if (conv_down)
        identity = &((*identity) >> conv_down);

    merge->forward(conv2->y(), *identity);

    merge->y() >> relu2;
}

void BasicBlock::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!dy.isEmpty());

    dy << relu2 << merge;

    auto *identity = &merge->dx();

    if (conv_down)
        identity = &((*identity) << conv_down);

    merge->dx() << conv2 << relu1 << conv1;

    split->backward(conv1->dx(), *identity);
}

LayerResNet18::LayerResNet18(const TensorGpu &x, Workspace &workspace)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;

    try {
        *this += new layer::LayerConvNorm(x, workspace, 64, 7, 7, 2, 3);
        *this += new layer::LayerActivationRelu(y());
        *this += new layer::LayerPooling(y(), 3, 2, 1);

        for (auto channel : {64, 128, 256, 512}) {
            *this += new BasicBlock(y(), workspace, channel, channel != 64);
            *this += new BasicBlock(y(), workspace, channel);
        }
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

} // namespace layer
} // namespace deepnet
