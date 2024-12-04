/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerActivationLeakyRelu.hpp"
#include "deepnet/Debug.hpp"

void gpu_leaky_relu(float *y, float *x, size_t n, float slope);

namespace deepnet {
    
namespace layer {

LayerActivationLeakyRelu::LayerActivationLeakyRelu(const TensorGpu &x,
                                                   float slope)
    : Layer(x), _slope(slope) {
    DEEPNET_TRACER;
    DEEPNET_ASSERT(_slope > 0.0f);

    _y.setDimension(x.dimension());
}

void LayerActivationLeakyRelu::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);
    gpu_leaky_relu(_y.data(), x.data(), x.size(), _slope);
};

void LayerActivationLeakyRelu::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    Layer::backward(dy);
    gpu_leaky_relu(_dx.data(), dy.data(), dy.size(), 1.0f / _slope);
};

void LayerActivationLeakyRelu::print(tool::TablePrinter &printer, //
                                     int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());

    printer.addRow({std::string(depth, '-') + std::to_string(index), //
                    std::string(type()),                             //
                    (std::string)output.dimension(),                 //
                    "",                                              //
                    std::string("slope=") + std::to_string(_slope)});
}
} // namespace layer
} // namespace deepnet
