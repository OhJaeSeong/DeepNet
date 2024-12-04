/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerFCNorm.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/Weight.hpp"
#include <sstream>

namespace deepnet {
namespace layer {

LayerFCNorm::~LayerFCNorm() {
    if (fc) {
        delete fc;
        fc = nullptr;
    }

    if (norm) {
        delete norm;
        norm = nullptr;
    }
}

void LayerFCNorm::train(void) {
    DEEPNET_ASSERT(fc);
    DEEPNET_ASSERT(norm);

    fc->train();
    norm->train();
}

void LayerFCNorm::eval(void) {
    DEEPNET_ASSERT(fc);
    DEEPNET_ASSERT(norm);

    fc->eval();
    norm->eval();
}

void LayerFCNorm::randomizeWeight(Weight::InitMethod method) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(fc);
    DEEPNET_ASSERT(norm);

    fc->randomizeWeight(method);
    norm->randomizeWeight(method);
}

void LayerFCNorm::readWeight(FILE *file, Weight::Format format) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(fc);
    DEEPNET_ASSERT(norm);

    if (format == Weight::Format::Darknet) {
        Weight::readWeight(file, norm->b, format);
        Weight::readWeight(file, norm->w, format);
        Weight::readWeight(file, norm->mean, format);
        Weight::readWeight(file, norm->variance, format);

        Weight::readWeight(file, fc->w, format);
    } else {
        Weight::readWeight(file, fc->w, format);

        Weight::readWeight(file, norm->w, format);
        Weight::readWeight(file, norm->b, format);
        Weight::readWeight(file, norm->mean, format);
        Weight::readWeight(file, norm->variance, format);
    }
}

void LayerFCNorm::writeWeight(FILE *file, Weight::Format format) const {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(fc);
    DEEPNET_ASSERT(norm);

    if (format == Weight::Format::Darknet) {
        Weight::writeWeight(file, norm->b, format);
        Weight::writeWeight(file, norm->w, format);
        Weight::writeWeight(file, norm->mean, format);
        Weight::writeWeight(file, norm->variance, format);

        Weight::writeWeight(file, fc->w, format);
    } else {
        Weight::writeWeight(file, fc->w, format);

        Weight::writeWeight(file, norm->w, format);
        Weight::writeWeight(file, norm->b, format);
        Weight::writeWeight(file, norm->mean, format);
        Weight::writeWeight(file, norm->variance, format);
    }
}

void LayerFCNorm::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(fc);
    DEEPNET_ASSERT(norm);

    x >> fc >> norm;
}

void LayerFCNorm::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(fc);
    DEEPNET_ASSERT(norm);

    dy << norm << fc;
}

/// 출력값을 반환한다.
const TensorGpu &LayerFCNorm::y(void) const {
    DEEPNET_ASSERT(norm);
    return norm->y();
};

/// 델타 x 값을 반환한다.
const TensorGpu &LayerFCNorm::dx(void) const {
    DEEPNET_ASSERT(fc);
    return fc->dx();
}

void LayerFCNorm::print(tool::TablePrinter &printer, //
                        int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = norm->y();
    auto &filter = fc->w;

    printer.addRow({std::string(depth, '-') + std::to_string(index), //
                    std::string(type()),                             //
                    (std::string)output.dimension(),                 //
                    (std::string)filter.dimension(),                 //
                    std::string("epsilon=") + std::to_string(_epsilon)});
}

void LayerFCNorm::debug(tool::TablePrinter &printer, int depth, int index) {
    DEEPNET_TRACER;

    TensorCpu output(norm->y());
    debugOutput(printer, output, depth, index);
}

void LayerFCNorm::update(float learning_rate, Weight::UpdateMethod method) {
    DEEPNET_ASSERT(fc);
    DEEPNET_ASSERT(norm);

    fc->update(learning_rate, method);
    norm->update(learning_rate, method);
}

unsigned char LayerFCNorm::checksum(void) const {
    auto checksum_fc = fc->checksum();
    auto checksum_norm = norm->checksum();

    return ~(~checksum_fc + ~checksum_norm);
}

} // namespace layer
} // namespace deepnet
