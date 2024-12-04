/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerMerge.hpp"
#include "deepnet/Debug.hpp"

namespace deepnet {
namespace layer {

LayerMerge::LayerMerge(const TensorGpu &x) : Layer(x), _op_desc(nullptr) {
    DEEPNET_TRACER;

    // Layer(x) 생성자를 사용하지 않는다(_dx를 사용하지 않기 때문).
    _px = &x;
    _training = false;

    SAFE_CUDNN(cudnnCreateOpTensorDescriptor(&_op_desc));

    SAFE_CUDNN(cudnnSetOpTensorDescriptor(      //
        _op_desc,                               //
        cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD, //
        cudnnDataType_t::CUDNN_DATA_FLOAT,      //
        cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN));

    _y.setDimension(x.dimension());
}

LayerMerge::~LayerMerge() {
    UNSAFE_CUDNN(cudnnDestroyOpTensorDescriptor(_op_desc));
}

void LayerMerge::forward(const TensorGpu &x1, const TensorGpu &x2) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(_op_desc);
    DEEPNET_ASSERT(!_y.isEmpty());
    auto y_dimension = _y.dimension();

    DEEPNET_ASSERT(x1.dimension() == y_dimension);
    DEEPNET_ASSERT(x2.dimension() == y_dimension);

    // Layer::forward(x1)를 사용하지 않는다.

    // y = x1 + x2 를 연산한다.
    float alpha = 1.0f;
    float beta = 0.0f;

    SAFE_CUDNN(cudnnOpTensor( //
        cudnn_handle,         //
        _op_desc,             //
        (const void *)&alpha, //
        x1.descriptor(),      //
        x1.data(),            //
        (const void *)&alpha, //
        x2.descriptor(),      //
        x2.data(),            //
        (const void *)&beta,  //
        _y.descriptor(),      //
        _y.data()));
}

void LayerMerge::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    _pdy = &dy;
    Layer::backward(dy);
}

} // namespace layer
} // namespace deepnet
