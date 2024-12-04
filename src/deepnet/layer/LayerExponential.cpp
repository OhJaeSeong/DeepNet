/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerExponential.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>

#define USE_CUDNN

// For CUDNN API,
// See https://docs.nvidia.com/deeplearning/cudnn/api/index.html

void gpu_exponential_forward(float *y, float *x, int channel, int width, int height, int group);

namespace deepnet {
namespace layer {

LayerExponential::LayerExponential(       //
    const TensorGpu &x, int group)
    : Layer(x), _group(group) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!x.isEmpty());

    _y.setDimension(x.batch(), int(x.channel()/group), 1, 1);
}

void LayerExponential::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    try {
        Layer::forward(x);
        gpu_exponential_forward(_y.data(), x.data(), x.channel(), x.width(), x.height(), _group);

    } catch (std::exception &e) {
        DEEPNET_LOG("Error = " << e.what());

        throw e;
    }
}

void LayerExponential::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT_MSG(false, "Not implemented!");
}

void LayerExponential::print(tool::TablePrinter &printer, //
                               int depth, int index) const {
    DEEPNET_TRACER;
}


} // namespace layer
} // namespace deepnet
