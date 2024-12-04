/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/Tensor.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/layer/Layer.hpp"

namespace deepnet {

void TensorGpu::resize(size_t new_size) {
    if (_size == new_size)
        return;

    // 이전 메모리를 삭제한다.
    if (_data)
        SAFE_CUDA(cudaFree((void *)_data));

    _size = new_size;

    if (_size == 0) {
        _data = nullptr;
        return;
    }

    // 새 메모리를 할당한다.
    SAFE_CUDA(cudaMalloc((void **)&_data, sizeof(float) * _size));
}

TensorGpu::TensorGpu(const TensorCpu &tensor)
    : TensorGpu(tensor.dimension()) {
    this->from(tensor);
}

void TensorGpu::setDimension(int batch, int channel, int height, int width) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(batch >= 0);
    DEEPNET_ASSERT(channel >= 0);
    DEEPNET_ASSERT(height >= 0);
    DEEPNET_ASSERT(width >= 0);

    _batch = batch;
    _channel = channel;
    _height = height;
    _width = width;

    resize((size_t)_batch * _channel * _height * _width);

    if (_batch > 0 && _channel > 0 && _height > 0 && _width > 0) {
        DEEPNET_ASSERT(_descriptor);

        SAFE_CUDNN(cudnnSetTensor4dDescriptor( //
            _descriptor,                       //
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, //
            _batch, _channel, //
            _height, _width));
    }
}

void TensorGpu::from(const TensorCpu &t) {
    DEEPNET_ASSERT(_size == t.size());

    if (_size > 0 && t.data() && _data)
        SAFE_CUDA(cudaMemcpy((void *)_data,         //
                             (void *)t.data(),      //
                             sizeof(float) * _size, //
                             cudaMemcpyKind::cudaMemcpyHostToDevice));
}

void TensorGpu::operator=(const TensorGpu &t) {
    DEEPNET_ASSERT(_size == t.size());

    if (_size > 0 && t.data() && _data)
        SAFE_CUDA(cudaMemcpy((void *)_data,         //
                             (void *)t.data(),      //
                             sizeof(float) * _size, //
                             cudaMemcpyKind::cudaMemcpyDeviceToDevice));
}

const TensorGpu &TensorGpu::operator>>(layer::Layer *l) const {
    l->forward(*this);

    return l->y();
}

const TensorGpu &TensorGpu::operator<<(layer::Layer *l) const {
    l->backward(*this);

    return l->dx();
}

} // namespace deepnet
