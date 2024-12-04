/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerDarknet19.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/layer/LayerActivationLeakyRelu.hpp"
#include "deepnet/layer/LayerBatchNorm.hpp"
#include "deepnet/layer/LayerConvolutional.hpp"
#include "deepnet/layer/LayerPooling.hpp"

namespace deepnet {
namespace layer {

void LayerDarknet19::conv(const TensorGpu &x, int channel, int size = 3) {
    *this += new LayerConvolutional(x, *_workspace,         //
                                    channel, size, size, 1, //
                                    (size == 1) ? 0 : 1);
    *this += new LayerBatchNorm(y());
    *this += new LayerActivationLeakyRelu(y(), 0.1f);
}

void LayerDarknet19::pool(const TensorGpu &x) {
    *this += new LayerPooling(x, 2, 2);
}

LayerDarknet19::LayerDarknet19(const TensorGpu &x, Workspace &workspace)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;

    // See https://github.com/uvipen/Yolo-v2-pytorch/blob/master/src/yolo_net.py

    try {
        conv(x, 32);       // stage1_conv1
        pool(y());         //
        conv(y(), 64);     // stage1_conv2
        pool(y());         //
        conv(y(), 128);    // stage1_conv3
        conv(y(), 64, 1);  // stage1_conv4
        conv(y(), 128);    // stage1_conv5
        pool(y());         //
        conv(y(), 256);    // stage1_conv6
        conv(y(), 128, 1); // stage1_conv7
        conv(y(), 256);    // stage1_conv8
        pool(y());         //
        conv(y(), 512);    // stage1_conv9
        conv(y(), 256, 1); // stage1_conv10
        conv(y(), 512);    // stage1_conv11
        conv(y(), 256, 1); // stage1_conv12
        conv(y(), 512);    // stage1_conv13
        pool(y());         // stage2_a_maxpl
        conv(y(), 1024);   // stage2_a_conv1
        conv(y(), 512, 1); // stage2_a_conv2
        conv(y(), 1024);   // stage2_a_conv3
        conv(y(), 512, 1); // stage2_a_conv4
        conv(y(), 1024);   // stage2_a_conv5
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

} // namespace layer
} // namespace deepnet
