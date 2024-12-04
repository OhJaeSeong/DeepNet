/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerNormalization.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/Weight.hpp"
#include <sstream>

namespace deepnet {
namespace layer {

LayerNormalization::LayerNormalization(const TensorGpu &x) : LayerWeighted(x) {
    DEEPNET_ASSERT(!x.isEmpty());

    auto dims = x.dimension();

    _y.setDimension(dims);

    dims = Dimension(1, dims.channel(), 1, 1);

    w.setDimension(dims);
    b.setDimension(dims);

    SAFE_CUDNN(cudnnCreateOpTensorDescriptor(&_op_desc_add));
    SAFE_CUDNN(cudnnCreateOpTensorDescriptor(&_op_desc_mul));

    SAFE_CUDNN(cudnnSetOpTensorDescriptor(      //
        _op_desc_add,                           //
        cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD, //
        cudnnDataType_t::CUDNN_DATA_FLOAT,      //
        cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN));

    SAFE_CUDNN(cudnnSetOpTensorDescriptor(      //
        _op_desc_mul,                           //
        cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MUL, //
        cudnnDataType_t::CUDNN_DATA_FLOAT,      //
        cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN));
}

LayerNormalization::~LayerNormalization() {
    UNSAFE_CUDNN(cudnnDestroyOpTensorDescriptor(_op_desc_add));
    UNSAFE_CUDNN(cudnnDestroyOpTensorDescriptor(_op_desc_mul));
}

void LayerNormalization::readWeight(FILE *file, Weight::Format format) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(format == Weight::Format::Torch);

    try {
        Weight::readWeight(file, b, format);
        Weight::readWeight(file, w, format);
    } catch (const std::exception &e) {
        DEEPNET_LOG("type = " << type());
        throw e;
    }

    TensorCpu w_cpu(w);
    auto size = w_cpu.size();
    auto data = w_cpu.data();

    // 모든 값을 역수로 변환한다.
    for (auto i = 0; i < size; i++, data++) {
        DEEPNET_ASSERT(*data != 0.0f);
        *data = 1.0f / *data;
    }

    w.from(w_cpu);
}

void LayerNormalization::writeWeight(FILE *file, Weight::Format format) const {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(format == Weight::Format::Torch);

    try {
        Weight::writeWeight(file, b, format);
        Weight::writeWeight(file, w, format);
    } catch (const std::exception &e) {
        DEEPNET_LOG("type = " << type());
        throw e;
    }
}

void LayerNormalization::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    LayerWeighted::forward(x);

    float plus = 1.0f;
    float minus = -1.0f;
    float zero = 0.0f;

    // 평균을 뺀다.
    // y = x - b.
    SAFE_CUDNN(cudnnOpTensor( //
        cudnn_handle,         //
        _op_desc_add,         //
        &plus,                //
        x.descriptor(),       //
        x.data(),             //
        &minus,               //
        b.descriptor(),       //
        b.data(),             //
        &zero,                //
        _y.descriptor(),      //
        _y.data()));

    // 표준편차로 나눈다.
    // y = y * w = y * (1 / std)
    SAFE_CUDNN(cudnnOpTensor( //
        cudnn_handle,         //
        _op_desc_mul,         //
        &plus,                //
        _y.descriptor(),      //
        _y.data(),            //
        &plus,                //
        w.descriptor(),       //
        w.data(),             //
        &zero,                //
        _y.descriptor(),      //
        _y.data()));
}

void LayerNormalization::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    LayerWeighted::backward(dy);
}

void LayerNormalization::print(tool::TablePrinter &printer, int depth,
                               int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());

    printer.addRow({std::string(depth, '-') + std::to_string(index), //
                    std::string(type()),                             //
                    (std::string)output.dimension(),                 //
                    "",                                              //
                    ""});
}

#define SS(x) std::to_string(x)

void LayerNormalization::printWeight(tool::TablePrinter &printer, int depth,
                                     int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());
    TensorCpu w_cpu(w);
    TensorCpu b_cpu(b);

    float w_min = 0.0f, w_max = 0.0f;
    std::tie(w_min, w_max) = w_cpu.getMinMax();
    float b_min = 0.0f, b_max = 0.0f;
    std::tie(b_min, b_max) = b_cpu.getMinMax();

    printer.addRow({std::string(depth, '-') + std::to_string(index),
                    std::string(type()),             //
                    (std::string)output.dimension(), //
                    std::string("w=") + SS(w_min) + "~" + SS(w_max) +
                        ", b=" + SS(b_min) + "~" + SS(b_max)});
}

unsigned char LayerNormalization::checksum(void) const {
    auto checksum_w = TensorCpu(w).checksum();
    auto checksum_b = TensorCpu(b).checksum();

    return ~(~checksum_w + ~checksum_b);
}

} // namespace layer
} // namespace deepnet
