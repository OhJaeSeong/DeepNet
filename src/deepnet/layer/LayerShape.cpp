/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerShape.hpp"
#include "deepnet/Debug.hpp"

namespace deepnet {
namespace layer {

LayerShape::LayerShape(const TensorGpu &x, Dimension dims) : Layer(x) {
    DEEPNET_TRACER;

    _y.setDimension(dims);

    DEEPNET_ASSERT(x.size() == _y.size());
}

void LayerShape::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);

    // 디바이스 메모리만 복사한다. (_y = x)
    SAFE_CUDA(cudaMemcpy((void *)_y.data(),         //
                         (void *)x.data(),          //
                         sizeof(float) * _y.size(), //
                         cudaMemcpyDeviceToDevice));
}

void LayerShape::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    Layer::backward(dy);

    // 디바이스 메모리만 복사한다. (_dx = dy)
    SAFE_CUDA(cudaMemcpy((void *)_dx.data(),         //
                         (void *)dy.data(),          //
                         sizeof(float) * _dx.size(), //
                         cudaMemcpyDeviceToDevice));
}

} // namespace layer
} // namespace deepnet
