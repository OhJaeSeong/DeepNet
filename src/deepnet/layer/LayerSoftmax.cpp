/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerSoftmax.hpp"
#include "deepnet/Debug.hpp"

namespace deepnet {
namespace layer {

LayerSoftmax::LayerSoftmax(const TensorGpu &x, int dim) : Layer(x) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!x.isEmpty());

    _y.setDimension(x.dimension());
    _dim = dim;
}

void LayerSoftmax::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);

    float alpha = 1.0f;
    float beta = 0.0f;
    if(_dim == 1){
        SAFE_CUDNN(cudnnSoftmaxForward(                      //
        cudnn_handle,                                    //
        cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE, //
        cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL, //
        &alpha,                                          //
        x.descriptor(),                                  //
        x.data(),                                      //
        &beta,                                           //
        _y.descriptor(),                                 //
        _y.data()));
    }else{
        SAFE_CUDNN(cudnnSoftmaxForward(                      //
        cudnn_handle,                                    //
        cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE, //
        cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE, //
        &alpha,                                          //
        x.descriptor(),                                  //
        x.data(),                                      //
        &beta,                                           //
        _y.descriptor(),                                 //
        _y.data()));
    }
    
}

void LayerSoftmax::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    Layer::backward(dy);

    float alpha = 1.0f;
    float beta = 0.0f;

    SAFE_CUDNN(cudnnSoftmaxBackward(                     //
        cudnn_handle,                                    //
        cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE,     //
        cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE, //
        &alpha,
        _y.descriptor(),  //
        _y.data(),      //
        dy.descriptor(),  //
        dy.data(),      //
        &beta,            //
        _dx.descriptor(), //
        _dx.data()));
}

} // namespace layer
} // namespace deepnet
