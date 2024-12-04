/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerVariance.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>

#define USE_CUDNN

// For CUDNN API,
// See https://docs.nvidia.com/deeplearning/cudnn/api/index.html

void gpu_variance_forward(float *y, float *x1, float *x2, 
                        int channel, int width, int height, int group);

namespace deepnet {
namespace layer {

LayerVariance::LayerVariance(       //
    const TensorGpu &x1, const TensorGpu &x2)
    : Layer(x1) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!x1.isEmpty());

    _y.setDimension(x2.batch(), x2.channel(), 1, 1);
}

void LayerVariance::forward(const TensorGpu &x1, const TensorGpu &x2) {
    DEEPNET_TRACER;

    try {
        Layer::forward(x1);
        int group = int(x1.channel()/x2.channel());
        gpu_variance_forward(_y.data(), x1.data(), x2.data(), x1.channel(), x1.width(), x1.height(), group);

    } catch (std::exception &e) {
        DEEPNET_LOG("Error = " << e.what());
        throw e;
    }
}

void LayerVariance::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT_MSG(false, "Not implemented!");
}

void LayerVariance::print(tool::TablePrinter &printer, //
                               int depth, int index) const {
    DEEPNET_TRACER;
}


} // namespace layer
} // namespace deepnet
