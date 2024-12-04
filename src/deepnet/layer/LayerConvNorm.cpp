/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerConvNorm.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/Weight.hpp"
#include <sstream>

namespace deepnet {
namespace layer {

LayerConvNorm::~LayerConvNorm() {
    if (conv) {
        delete conv;
        conv = nullptr;
    }

    if (norm) {
        delete norm;
        norm = nullptr;
    }
}

void LayerConvNorm::train(void) {
    DEEPNET_ASSERT(conv);
    DEEPNET_ASSERT(norm);

    conv->train();
    norm->train();
}

void LayerConvNorm::eval(void) {
    DEEPNET_ASSERT(conv);
    DEEPNET_ASSERT(norm);

    conv->eval();
    norm->eval();
}

void LayerConvNorm::randomizeWeight(Weight::InitMethod method) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(conv);
    DEEPNET_ASSERT(norm);

    conv->randomizeWeight(method);
    norm->randomizeWeight(method);
}

void LayerConvNorm::readWeight(FILE *file, Weight::Format format) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(conv);
    DEEPNET_ASSERT(norm);

    if (format == Weight::Format::Darknet) {
        Weight::readWeight(file, norm->b, format);
        Weight::readWeight(file, norm->w, format);
        Weight::readWeight(file, norm->mean, format);
        Weight::readWeight(file, norm->variance, format);

        Weight::readWeight(file, conv->w, format);
    } else {
        Weight::readWeight(file, conv->w, format);

        Weight::readWeight(file, norm->w, format);
        Weight::readWeight(file, norm->b, format);
        Weight::readWeight(file, norm->mean, format);
        Weight::readWeight(file, norm->variance, format);
    }
}

void LayerConvNorm::writeWeight(FILE *file, Weight::Format format) const {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(conv);
    DEEPNET_ASSERT(norm);

    if (format == Weight::Format::Darknet) {
        Weight::writeWeight(file, norm->b, format);
        Weight::writeWeight(file, norm->w, format);
        Weight::writeWeight(file, norm->mean, format);
        Weight::writeWeight(file, norm->variance, format);

        Weight::writeWeight(file, conv->w, format);
    } else {
        Weight::writeWeight(file, conv->w, format);

        Weight::writeWeight(file, norm->w, format);
        Weight::writeWeight(file, norm->b, format);
        Weight::writeWeight(file, norm->mean, format);
        Weight::writeWeight(file, norm->variance, format);
    }
}

void LayerConvNorm::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(conv);
    DEEPNET_ASSERT(norm);

    x >> conv >> norm;
}

void LayerConvNorm::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(conv);
    DEEPNET_ASSERT(norm);

    dy << norm << conv;
}

/// 출력값을 반환한다.
const TensorGpu &LayerConvNorm::y(void) const {
    DEEPNET_ASSERT(norm);
    return norm->y();
};

/// 델타 x 값을 반환한다.
const TensorGpu &LayerConvNorm::dx(void) const {
    DEEPNET_ASSERT(conv);
    return conv->dx();
}

void LayerConvNorm::print(tool::TablePrinter &printer, //
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

#define SS(x) std::to_string(x)

void LayerConvNorm::printWeight(tool::TablePrinter &printer, int depth,
                                int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());

    DEEPNET_ASSERT(conv);
    DEEPNET_ASSERT(norm);

    TensorCpu w_cpu(conv->w);
    TensorCpu scale(norm->w);
    TensorCpu bias(norm->b);
    TensorCpu mean(norm->mean);
    TensorCpu variance(norm->variance);

    float w_min = 0.0f, w_max = 0.0f;
    std::tie(w_min, w_max) = w_cpu.getMinMax();

    float scale_min = 0.0f, scale_max = 0.0f;
    std::tie(scale_min, scale_max) = scale.getMinMax();

    float bias_min = 0.0f, bias_max = 0.0f;
    std::tie(bias_min, bias_max) = bias.getMinMax();

    float mean_min = 0.0f, mean_max = 0.0f;
    std::tie(mean_min, mean_max) = mean.getMinMax();

    float variance_min = 0.0f, variance_max = 0.0f;
    std::tie(variance_min, variance_max) = variance.getMinMax();

    printer.addRow({std::string(depth, '-') + std::to_string(index),
                    std::string(type()), //
                    (std::string)output.dimension(),
                    std::string("w=") + SS(w_min) + "~" + SS(w_max) +  //
                        ", s=" + SS(scale_min) + "~" + SS(scale_max) + //
                        ", b=" + SS(bias_min) + "~" + SS(bias_max) +   //
                        ", u=" + SS(mean_min) + "~" + SS(mean_max) +   //
                        ", v=" + SS(variance_min) + "~" + SS(variance_max)});
}

void LayerConvNorm::debug(tool::TablePrinter &printer, int depth, int index) {
    DEEPNET_TRACER;

    TensorCpu output(norm->y());
    debugOutput(printer, output, depth, index);
}

void LayerConvNorm::update(float learning_rate, Weight::UpdateMethod method) {
    DEEPNET_ASSERT(conv);
    DEEPNET_ASSERT(norm);

    conv->update(learning_rate, method);
    norm->update(learning_rate, method);
}

unsigned char LayerConvNorm::checksum(void) const {
    auto checksum_conv = conv->checksum();
    auto checksum_norm = norm->checksum();

    return ~(~checksum_conv + ~checksum_norm);
}

} // namespace layer
} // namespace deepnet
