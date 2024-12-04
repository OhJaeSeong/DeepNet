/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerConvNormLeaky.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/Weight.hpp"
#include <sstream>

namespace deepnet {
namespace layer {

LayerConvNormLeaky::~LayerConvNormLeaky() {
    if (conv) {
        delete conv;
        conv = nullptr;
    }

    if (norm) {
        delete norm;
        norm = nullptr;
    }

    if (leaky) {
        delete leaky;
        leaky = nullptr;
    }
}

void LayerConvNormLeaky::train(void) {
    LayerConvNorm::train();

    DEEPNET_ASSERT(leaky);
    leaky->train();
}

void LayerConvNormLeaky::eval(void) {
    LayerConvNorm::train();

    DEEPNET_ASSERT(leaky);
    leaky->eval();
}

void LayerConvNormLeaky::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    x >> conv >> norm >> leaky;
}

void LayerConvNormLeaky::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    dy << leaky << norm << conv;
}

/// 출력값을 반환한다.
const TensorGpu &LayerConvNormLeaky::y(void) const {
    DEEPNET_ASSERT(leaky);
    return leaky->y();
};

/// 델타 x 값을 반환한다.
const TensorGpu &LayerConvNormLeaky::dx(void) const {
    DEEPNET_ASSERT(conv);
    return conv->dx();
}

void LayerConvNormLeaky::print(tool::TablePrinter &printer, //
                               int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = norm->y();
    auto &filter = conv->w;

    printer.addRow({std::string(depth, '-') + std::to_string(index),        //
                    std::string(type()),                                    //
                    (std::string)output.dimension(),                        //
                    (std::string)filter.dimension(),                        //
                    std::string("stride=") + std::to_string(_stride) + ", " //
                        + "padding=" + std::to_string(_padding) + ", "      //
                        + "epsilon=" + std::to_string(_epsilon) + ", "      //
                        + "slope=" + std::to_string(_slope)});
}

void LayerConvNormLeaky::debug(tool::TablePrinter &printer, int depth,
                               int index) {
    DEEPNET_TRACER;

    TensorCpu output(leaky->y());
    debugOutput(printer, output, depth, index);
}

} // namespace layer
} // namespace deepnet
