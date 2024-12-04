/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerActivation.hpp"
#include "deepnet/Debug.hpp"

namespace deepnet {
namespace layer {

LayerActivation::LayerActivation(const TensorGpu &x)
    : Layer(x), _act_desc(nullptr) {
    DEEPNET_TRACER;

    SAFE_CUDNN(cudnnCreateActivationDescriptor(&_act_desc));

    _y.setDimension(x.dimension());
}

LayerActivation::~LayerActivation() {
    if (_act_desc) {
        cudnnDestroyActivationDescriptor(_act_desc);
        _act_desc = nullptr;
    }
}

void LayerActivation::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(_act_desc);

    Layer::forward(x);

    float alpha = 1.0f;
    float beta = 0.0f;

    SAFE_CUDNN(cudnnActivationForward( //
        cudnn_handle,                  //
        _act_desc,                     //
        (const void *)&alpha,          //
        x.descriptor(),                //
        x.data(),                      //
        (const void *)&beta,           //
        _y.descriptor(),               //
        _y.data()));
}

void LayerActivation::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(_act_desc);

    Layer::backward(dy);

    float alpha = 1.0f;
    float beta = 0.0f;

    SAFE_CUDNN(cudnnActivationBackward( //
        cudnn_handle,                   //
        _act_desc,                      //
        (const void *)&alpha,           //
        _y.descriptor(),                //
        _y.data(),                      //
        dy.descriptor(),                //
        dy.data(),                    //
        _px->descriptor(),              //
        _px->data(),                  //
        (const void *)&beta,            //
        _dx.descriptor(),               //
        _dx.data()));
}

} // namespace layer
} // namespace deepnet
