/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/Layer.hpp"
#include <sstream>

namespace deepnet {
namespace layer {

Layer::Layer(const TensorGpu &x) : _px(&x), _training(false) {
    DEEPNET_ASSERT(cudnn_handle);
}

void Layer::train(void) {
    DEEPNET_ASSERT(_px);
    DEEPNET_ASSERT(!_px->isEmpty());

    _training = true;
    _dx.setDimension(_px->dimension());
}

void Layer::eval(void) {
    _training = false;
    _dx.setDimension(0, 0, 0, 0);
}

void Layer::forward(const TensorGpu &x) {
    DEEPNET_ASSERT(_px && !_px->isEmpty());
    DEEPNET_ASSERT(!x.isEmpty());
    DEEPNET_ASSERT(_px->dimension() == x.dimension());

    // 입력값을 기억한다.
    _px = &x;
}

void Layer::backward(const TensorGpu &dy) {
    DEEPNET_ASSERT(_training);
    DEEPNET_ASSERT(_px);
    DEEPNET_ASSERT(!_px->isEmpty());
    DEEPNET_ASSERT(!dy.isEmpty());

    if (!(y().dimension() == dy.dimension())) {
        DEEPNET_LOG("y.dimension() = " << (std::string)y().dimension());
        DEEPNET_LOG("dy.dimension() = " << (std::string)dy.dimension());

        DEEPNET_ASSERT(y().dimension() == dy.dimension());
    }
}

const TensorGpu &Layer::y(void) const { return _y; }

const TensorGpu &Layer::dx(void) const { return _dx; }

static std::string align(std::string value, int size) {
    int space_length = size - (int)value.size();

    if (space_length <= 0)
        return value;

    std::string space;
    space.resize(space_length, ' ');

    return value + space;
}

void Layer::print(tool::TablePrinter &printer, int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());

    printer.addRow({std::string(depth, '-') + std::to_string(index),
                    std::string(type()),             //
                    (std::string)output.dimension(), //
                    "", ""});
}

void Layer::printWeight(tool::TablePrinter &printer, int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());

    printer.addRow({std::string(depth, '-') + std::to_string(index),
                    std::string(type()),             //
                    (std::string)output.dimension(), //
                    ""});
}

void Layer::debug(tool::TablePrinter &printer, int depth, int index) {
    DEEPNET_TRACER;

    TensorCpu output(y());
    debugOutput(printer, output, depth, index);
}

void Layer::debugOutput(tool::TablePrinter &printer, TensorCpu &output, //
                        int depth, int index) {
    DEEPNET_TRACER;

    float min = 0.0f, max = 0.0f;
    std::tie(min, max) = output.getMinMax();

    float d1, d2, d3;
    d1 = output[0];
    if (output.size() > 1)
        d2 = output[1];
    if (output.size() > 2)
        d3 = output[2];

    auto data = std::to_string(d1) + ", "   //
                + std::to_string(d2) + ", " //
                + std::to_string(d3);

    printer.addRow({std::string(depth, '-') + std::to_string(index),
                    std::string(type()),             //
                    (std::string)output.dimension(), //
                    std::to_string(min),             //
                    std::to_string(max),             //
                    data});
}

} // namespace layer
} // namespace deepnet
