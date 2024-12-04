/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerActivationElu.hpp"
#include "deepnet/Debug.hpp"

namespace deepnet {
namespace layer {

LayerActivationElu::LayerActivationElu(const TensorGpu &x) : LayerActivation(x) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(_act_desc);

    SAFE_CUDNN(cudnnSetActivationDescriptor( //
        _act_desc,                           //
        CUDNN_ACTIVATION_ELU,                //
        CUDNN_PROPAGATE_NAN,                 //
        0.0));
}

} // namespace layer
} // namespace deepnet
