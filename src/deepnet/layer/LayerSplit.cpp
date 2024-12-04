/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerSplit.hpp"
#include "deepnet/Debug.hpp"

namespace deepnet {
namespace layer {

LayerSplit::LayerSplit(const TensorGpu &x) : Layer(x), _op_desc(nullptr) {
    DEEPNET_TRACER;

    SAFE_CUDNN(cudnnCreateOpTensorDescriptor(&_op_desc));

    SAFE_CUDNN(cudnnSetOpTensorDescriptor(      //
        _op_desc,                               //
        cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD, //
        cudnnDataType_t::CUDNN_DATA_FLOAT,      //
        cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN));
}

LayerSplit::~LayerSplit() {
    UNSAFE_CUDNN(cudnnDestroyOpTensorDescriptor(_op_desc));
}

void LayerSplit::backward(const TensorGpu &dy1, const TensorGpu &dy2) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(_op_desc);

    Layer::backward(dy1);

    DEEPNET_ASSERT(_dx.dimension() == dy2.dimension());

    // _dx = dy1 + dy2 를 연산한다.
    float alpha = 1.0f;
    float beta = 0.0f;

    SAFE_CUDNN(cudnnOpTensor( //
        cudnn_handle,         //
        _op_desc,             //
        (const void *)&alpha, //
        dy1.descriptor(),     //
        dy1.data(),           //
        (const void *)&alpha, //
        dy2.descriptor(),     //
        dy2.data(),           //
        (const void *)&beta,  //
        _dx.descriptor(),     //
        _dx.data()));
}

} // namespace layer
} // namespace deepnet
