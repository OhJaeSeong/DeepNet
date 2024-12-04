/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerUpSample.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>

namespace deepnet {
namespace layer {

LayerUpSample::LayerUpSample(const TensorGpu &x, int sample)
    : Layer(x), _sample(sample), _pool_desc(nullptr) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!x.isEmpty());
    DEEPNET_ASSERT(sample >= 1);

    SAFE_CUDNN(cudnnCreatePoolingDescriptor(&_pool_desc));

    SAFE_CUDNN(cudnnSetPooling2dDescriptor(                              //
        _pool_desc,                                                      //
        cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, //
        cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,                      //
        _sample, _sample, 0, 0, _sample, _sample));

    _y.setDimension(x.batch(), x.channel(), x.height() * sample,
                    x.width() * sample);
}

LayerUpSample::~LayerUpSample() {
    if (_pool_desc)
        UNSAFE_CUDNN(cudnnDestroyPoolingDescriptor(_pool_desc));
}

void LayerUpSample::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);

    try {
        float alpha = (float)(_sample * _sample);
        float beta = 0.0f;

        SAFE_CUDNN(cudnnPoolingBackward( //
            cudnn_handle,                //
            _pool_desc,                  //
            &alpha,                      //
            x.descriptor(),              //
            x.data(),                    //
            x.descriptor(),              //
            x.data(),                    //
            _y.descriptor(),             //
            _y.data(),                   //
            &beta,                       //
            _y.descriptor(),             //
            _y.data()));
    } catch (std::exception &e) {
        DEEPNET_LOG("x= " << (std::string)x.dimension());

        throw e;
    }
};

void LayerUpSample::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    Layer::backward(dy);

    float alpha = 1.0f;
    float beta = 0.0f;

    SAFE_CUDNN(cudnnPoolingForward( //
        cudnn_handle,               //
        _pool_desc,                 //
        &alpha,                     //
        dy.descriptor(),            //
        dy.data(),                  //
        &beta,                      //
        _dx.descriptor(),           //
        _dx.data()));
};

void LayerUpSample::print(tool::TablePrinter &printer, //
                          int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = y();

    printer.addRow({std::string(depth, '-') + std::to_string(index), //
                    std::string(type()),                             //
                    (std::string)output.dimension(),                 //
                    "",                                              //
                    std::string("sample=") + std::to_string(_sample)});
}

} // namespace layer
} // namespace deepnet
