/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerActivationTanh.hpp"
#include "deepnet/Debug.hpp"

namespace deepnet {
namespace layer {

LayerActivationTanh::LayerActivationTanh(const TensorGpu &x) : LayerActivation(x) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(_act_desc);

    SAFE_CUDNN(cudnnSetActivationDescriptor( //
        _act_desc,                           //
        CUDNN_ACTIVATION_TANH,               //
        CUDNN_PROPAGATE_NAN,                 //
        0.0));
}

} // namespace layer
} // namespace deepnet
