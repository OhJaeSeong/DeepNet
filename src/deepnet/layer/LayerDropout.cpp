/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerDropout.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>

namespace deepnet {
namespace layer {

LayerDropout::LayerDropout(const TensorGpu &x, float dropout)
    : Layer(x), _dropout(dropout), _descriptor(nullptr), _state_size(0),
      _states(nullptr), _space_size(0), _spaces(nullptr) {
    DEEPNET_ASSERT(!x.isEmpty());

    _y.setDimension(x.dimension());

    SAFE_CUDNN(cudnnCreateDropoutDescriptor(&_descriptor));

    // 드롭된 가중치의 위치 정보를 할당한다.
    SAFE_CUDNN(cudnnDropoutGetStatesSize(cudnn_handle, &_state_size));

    SAFE_CUDA(cudaMalloc((void **)&_states, _state_size));

    // 드롭된 가중치의 값 정보를 할당한다.
    SAFE_CUDNN(cudnnDropoutGetReserveSpaceSize(x.descriptor(), &_space_size));

    SAFE_CUDA(cudaMalloc((void **)&_spaces, _space_size));

    SAFE_CUDNN(cudnnSetDropoutDescriptor( //
        _descriptor,                      //
        cudnn_handle,                     //
        _dropout,                         //
        _states,                          //
        _state_size,                      //
        0));
}

LayerDropout::~LayerDropout() {
    if (_descriptor) {
        UNSAFE_CUDNN(cudnnDestroyDropoutDescriptor(_descriptor));
        _descriptor = nullptr;
    }

    if (_states) {
        UNSAFE_CUDA(cudaFree((void *)_states));
        _states = nullptr;
    }

    if (_spaces) {
        UNSAFE_CUDA(cudaFree((void *)_spaces));
        _spaces = nullptr;
    }
}

void LayerDropout::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);

    float alpha = 1.0f;
    float beta = 0.0f;

    SAFE_CUDNN(cudnnDropoutForward( //
        cudnn_handle,               //
        _descriptor,                //
        x.descriptor(),             //
        x.data(),                   //
        _y.descriptor(),            //
        _y.data(),                  //
        _spaces,                    //
        _space_size));
}

void LayerDropout::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    Layer::backward(dy);

    float alpha = 1.0f;
    float beta = 0.0f;

    SAFE_CUDNN(cudnnDropoutBackward( //
        cudnn_handle,                //
        _descriptor,                 //
        dy.descriptor(),             //
        dy.data(),                   //
        _dx.descriptor(),            //
        _dx.data(),                  //
        _spaces,                     //
        _space_size));
}

void LayerDropout::print(tool::TablePrinter &printer, //
                         int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = y();

    printer.addRow({std::string(depth, '-') + std::to_string(index), //
                    std::string(type()),                             //
                    (std::string)output.dimension(),                 //
                    "",                                              //
                    std::string("ratio=") + std::to_string(_dropout)});
}

} // namespace layer
} // namespace deepnet
