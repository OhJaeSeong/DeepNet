/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerWeighted.hpp"

namespace deepnet {
namespace layer {

LayerWeighted::~LayerWeighted() {
    DEEPNET_TRACER;

    if (_w_desc)
        UNSAFE_CUDNN(cudnnDestroyFilterDescriptor(_w_desc));
    if (_conv_desc)
        UNSAFE_CUDNN(cudnnDestroyConvolutionDescriptor(_conv_desc));
}

void LayerWeighted::randomizeWeight(Weight::InitMethod method) {
    DEEPNET_TRACER;

    Weight::initializeWeight(w, method);

    if (_bias)
        Weight::initializeWeight(b, method);
}

void LayerWeighted::readWeight(FILE *file, Weight::Format format) {
    DEEPNET_TRACER;

    try {
        if (format == Weight::Format::Darknet) {
            if (_bias)
                Weight::readWeight(file, b, format);
            Weight::readWeight(file, w, format);
        } else {
            Weight::readWeight(file, w, format);
            if (_bias)
                Weight::readWeight(file, b, format);
        }
    } catch (const std::exception &e) {
        DEEPNET_LOG("type = " << type());
        throw e;
    }
}

void LayerWeighted::writeWeight(FILE *file, Weight::Format format) const {
    DEEPNET_TRACER;

    try {
        if (format == Weight::Format::Darknet) {
            if (_bias)
                Weight::writeWeight(file, b, format);
            Weight::writeWeight(file, w, format);
        } else {
            Weight::writeWeight(file, w, format);
            if (_bias)
                Weight::writeWeight(file, b, format);
        }
    } catch (const std::exception &e) {
        DEEPNET_LOG("type = " << type());
        throw e;
    }
}

void LayerWeighted::printWeight(tool::TablePrinter &printer, int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());

    printer.addRow({std::string(depth, '-') + std::to_string(index),
                    std::string(type()),             //
                    (std::string)output.dimension(), //
                    "??"});
}

void LayerWeighted::update(float learning_rate, Weight::UpdateMethod method) {

    float alpha = -learning_rate;

    // w -= lr * dw.
    SAFE_CUBLAS(cublasSaxpy( //
        cublas_handle,       //
        (int)w.size(),       //
        &alpha,              //
        dw.data(), 1,        //
        w.data(), 1));

    // b -= lr * db.
    SAFE_CUBLAS(cublasSaxpy( //
        cublas_handle,       //
        (int)b.size(),       //
        &alpha,              //
        db.data(), 1,        //
        b.data(), 1));
}

unsigned char LayerWeighted::checksum(void) const {
    auto checksum_w = TensorCpu(w).checksum();

    if (!_bias)
        return checksum_w;

    auto checksum_b = TensorCpu(b).checksum();

    return ~(~checksum_w + ~checksum_b);
}

} // namespace layer
} // namespace deepnet
