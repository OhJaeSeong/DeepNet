/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerPooling.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>

namespace deepnet {
namespace layer {

LayerPooling::LayerPooling(const TensorGpu &x, int size_height, int size_width,
                           int stride_height, int stride_width,
                           int padding_height, int padding_width, bool max)
    : Layer(x), _size_height(size_height), _size_width(size_width),
      _stride_height(stride_height), _stride_width(stride_width),
      _padding_height(padding_height), _padding_width(padding_width), _max(max),
      _pool_desc(nullptr) {
    DEEPNET_TRACER;

    if (size_height == 0)
        _size_height = size_height = x.height();

    if (size_width == 0)
        _size_width = size_width = x.width();

    DEEPNET_ASSERT(!x.isEmpty());
    DEEPNET_ASSERT(size_height >= 1);
    DEEPNET_ASSERT(size_width >= 1);
    DEEPNET_ASSERT(stride_height >= 1);
    DEEPNET_ASSERT(stride_width >= 1);
    DEEPNET_ASSERT(size_height >= stride_height);
    DEEPNET_ASSERT(size_width >= stride_width);

    SAFE_CUDNN(cudnnCreatePoolingDescriptor(&_pool_desc));

    SAFE_CUDNN(cudnnSetPooling2dDescriptor( //
        _pool_desc,                         //
        max ? cudnnPoolingMode_t::CUDNN_POOLING_MAX
            : cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
        cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN, //
        _size_height, _size_width,                  //
        _padding_height, _padding_width,            //
        _stride_height, _stride_width));

    auto height =
        (x.height() - _size_height + _padding_height * 2) /
        _stride_height + 1;
    auto width = (x.width() - _size_width + _padding_width * 2) /
                 _stride_width + 1;

    _y.setDimension(x.batch(), x.channel(), height, width);
}

LayerPooling::~LayerPooling() {
    if (_pool_desc)
        UNSAFE_CUDNN(cudnnDestroyPoolingDescriptor(_pool_desc));
}

void LayerPooling::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);

    float alpha = 1.0f;
    float beta = 0.0f;

    SAFE_CUDNN(cudnnPoolingForward( //
        cudnn_handle,               //
        _pool_desc,                 //
        &alpha,                     //
        x.descriptor(),             //
        x.data(),                   //
        &beta,                      //
        _y.descriptor(),            //
        _y.data()));
};

void LayerPooling::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    Layer::backward(dy);

    try {
        float alpha = 1.0f;
        float beta = 0.0f;

        SAFE_CUDNN(cudnnPoolingBackward( //
            cudnn_handle,                //
            _pool_desc,                  //
            &alpha,                      //
            _y.descriptor(),             //
            _y.data(),                   //
            dy.descriptor(),             //
            dy.data(),                   //
            _px->descriptor(),           //
            _px->data(),                 //
            &beta,                       //
            _dx.descriptor(),            //
            _dx.data()));
    } catch (std::exception &e) {
        DEEPNET_LOG("dx=" << (std::string)_dx.dimension()          //
                       << ", y=" << (std::string)_y.dimension() //
                       << ", dy=" << (std::string)dy.dimension());

        throw e;
    }
};

void LayerPooling::print(tool::TablePrinter &printer, //
                         int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = y();
    std::stringstream option;

    if (_size_height == _size_width        //
        && _stride_height == _stride_width //
        && _padding_height == _padding_width)
        option << "size="                 //
               << _size_height << ", "    //
               << "stride="               //
               << _stride_height << ", "  //
               << "padding="              //
               << _padding_height << ", " //
               << "op="                   //
               << (_max ? "max" : "avg");
    else
        option << "size=("                //
               << _size_height << ", "    //
               << _size_width << "), "    //
               << "stride=("              //
               << _stride_height << ", "  //
               << _stride_width << "), "  //
               << "padding=("             //
               << _padding_height << ", " //
               << _padding_width << "), " //
               << "op="                   //
               << (_max ? "max" : "avg");

    printer.addRow({std::string(depth, '-') + std::to_string(index), //
                    std::string(type()),                             //
                    (std::string)output.dimension(),                 //
                    "",                                              //
                    option.str()});
}

} // namespace layer
} // namespace deepnet
