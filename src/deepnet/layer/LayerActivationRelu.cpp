/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerActivationRelu.hpp"
#include "deepnet/Debug.hpp"

namespace deepnet {
namespace layer {

LayerActivationRelu::LayerActivationRelu(const TensorGpu &x) : LayerActivation(x) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(_act_desc);

    SAFE_CUDNN(cudnnSetActivationDescriptor( //
        _act_desc,                           //
        CUDNN_ACTIVATION_RELU,               //
        CUDNN_PROPAGATE_NAN,                 //
        0.0));
}

} // namespace layer
} // namespace deepnet
