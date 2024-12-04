/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerConvNormSilu.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/Weight.hpp"
#include <sstream>

namespace deepnet {
namespace layer {

LayerConvNormSilu::~LayerConvNormSilu() {
    if (conv) {
        delete conv;
        conv = nullptr;
    }

    if (norm) {
        delete norm;
        norm = nullptr;
    }

    if (silu) {
        delete silu;
        silu = nullptr;
    }
}

void LayerConvNormSilu::train(void) {
    LayerConvNorm::train();

    DEEPNET_ASSERT(silu);
    silu->train();
}

void LayerConvNormSilu::eval(void) {
    LayerConvNorm::train();

    DEEPNET_ASSERT(silu);
    silu->eval();
}

void LayerConvNormSilu::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    x >> conv >> norm >> silu;
}

void LayerConvNormSilu::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    dy << silu << norm << conv;
}

/// 출력값을 반환한다.
const TensorGpu &LayerConvNormSilu::y(void) const {
    DEEPNET_ASSERT(silu);
    return silu->y();
};

/// 델타 x 값을 반환한다.
const TensorGpu &LayerConvNormSilu::dx(void) const {
    DEEPNET_ASSERT(conv);
    return conv->dx();
}

void LayerConvNormSilu::print(tool::TablePrinter &printer, //
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

void LayerConvNormSilu::debug(tool::TablePrinter &printer, int depth,
                               int index) {
    DEEPNET_TRACER;

    TensorCpu output(silu->y());
    debugOutput(printer, output, depth, index);
}

} // namespace layer
} // namespace deepnet
