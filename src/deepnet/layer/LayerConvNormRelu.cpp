/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerConvNormRelu.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/Weight.hpp"
#include <sstream>

namespace deepnet {
namespace layer {

LayerConvNormRelu::~LayerConvNormRelu() {
    if (conv) {
        delete conv;
        conv = nullptr;
    }

    if (norm) {
        delete norm;
        norm = nullptr;
    }

    if (relu) {
        delete relu;
        relu = nullptr;
    }
}

void LayerConvNormRelu::train(void) {
    LayerConvNorm::train();

    DEEPNET_ASSERT(relu);
    relu->train();
}

void LayerConvNormRelu::eval(void) {
    LayerConvNorm::train();

    DEEPNET_ASSERT(relu);
    relu->eval();
}

void LayerConvNormRelu::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    x >> conv >> norm >> relu;
}

void LayerConvNormRelu::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    dy << relu << norm << conv;
}

/// 출력값을 반환한다.
const TensorGpu &LayerConvNormRelu::y(void) const {
    DEEPNET_ASSERT(relu);
    return relu->y();
};

/// 델타 x 값을 반환한다.
const TensorGpu &LayerConvNormRelu::dx(void) const {
    DEEPNET_ASSERT(conv);
    return conv->dx();
}

void LayerConvNormRelu::print(tool::TablePrinter &printer, //
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
                        + "epsilon=" + std::to_string(_epsilon)});
}

void LayerConvNormRelu::debug(tool::TablePrinter &printer, int depth,
                               int index) {
    DEEPNET_TRACER;

    TensorCpu output(relu->y());
    debugOutput(printer, output, depth, index);
}

} // namespace layer
} // namespace deepnet