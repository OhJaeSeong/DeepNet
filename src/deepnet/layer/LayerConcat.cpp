/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerConcat.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>

void gpu_concat_forward(float *y, float *x1, float *x2,        //
                        int batch, int channel1, int channel2, //
                        int height, int width);

void gpu_concat_backward(float *y, float *x1, float *x2,        //
                         int batch, int channel1, int channel2, //
                         int height, int width);

namespace deepnet {
namespace layer {

LayerConcat::LayerConcat(const TensorGpu &x1, const TensorGpu &x2)
    : Layer(x1), _px2(&x2) {

    DEEPNET_TRACER;

    _dx2.setDimension(x2.dimension());

    Dimension d1 = x1.dimension();
    Dimension d2 = x2.dimension();

    DEEPNET_ASSERT(d1.batch() == d2.batch());
    DEEPNET_ASSERT(d1.height() == d2.height());
    DEEPNET_ASSERT(d1.width() == d2.width());

    _y.setDimension(d1.batch(), d1.channel() + d2.channel(), d1.height(),
                    d1.width());
}

LayerConcat::~LayerConcat() {}

void LayerConcat::forward(const TensorGpu &x1, const TensorGpu &x2) {
    DEEPNET_TRACER;

    Layer::forward(x1);

    DEEPNET_ASSERT(x2.dimension() == _dx2.dimension());
    _px2 = &x2;

    gpu_concat_forward(_y.data(), x1.data(), x2.data(),        //
                       x1.batch(), x1.channel(), x2.channel(), //
                       x1.height(), x1.width());
}

void LayerConcat::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    Layer::backward(dy);

    gpu_concat_backward(dy.data(), _dx.data(), _dx2.data(),         //
                        _dx.batch(), _dx.channel(), _dx2.channel(), //
                        _dx.height(), _dx.width());
}

inline TensorGpu &LayerConcat::dx2(void) { return _dx2; }

void LayerConcat::print(tool::TablePrinter &printer, //
                        int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = y();

    printer.addRow({std::string(depth, '-') + std::to_string(index), //
                    std::string(type()),                             //
                    (std::string)output.dimension(),                 //
                    "",                                              //
                    std::string("channel=(") + std::to_string(_px->channel()) +
                        ", " //
                        + std::to_string(_px2->channel()) + ")"});
}

} // namespace layer
} // namespace deepnet
