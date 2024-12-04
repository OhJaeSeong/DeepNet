/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerFlatten.hpp"
#include "deepnet/Debug.hpp"

namespace deepnet {
namespace layer {

/// 생성자.
LayerFlatten::LayerFlatten(const TensorGpu &x, int dim) : Layer(x) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!x.isEmpty());
    if(dim == 1){
        _y.setDimension(x.batch(), x.channel() * x.height() * x.width(), 1, 1);
    }else if(dim == 2){
        _y.setDimension(x.batch(), x.channel(), x.height() * x.width(), 1);
    }  
}

void LayerFlatten::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);

    SAFE_CUDA(cudaMemcpy((void *)_y.data(),         //
                         (void *)x.data(),          //
                         sizeof(float) * _y.size(), //
                         cudaMemcpyKind::cudaMemcpyDeviceToDevice));
}

void LayerFlatten::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    Layer::backward(dy);

    SAFE_CUDA(cudaMemcpy((void *)_dx.data(),         //
                         (void *)dy.data(),          //
                         sizeof(float) * _dx.size(), //
                         cudaMemcpyKind::cudaMemcpyDeviceToDevice));
}

} // namespace layer
} // namespace deepnet
