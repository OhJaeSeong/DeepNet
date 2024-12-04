/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerCat.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>

void gpu_cat_forward(float *y, float *x1, float *x2,        //
                        int batch, int channel,
                        int channel1, int height2, int width);

void gpu_cat_backward(float *y, float *x1, float *x2,        //
                        int batch, int channel,
                        int height1, int height2, int width);

namespace deepnet {
namespace layer {

LayerCat::LayerCat(const TensorGpu &x1, const TensorGpu &x2)
    : Layer(x1), _px2(&x2) {

    DEEPNET_TRACER;

    _dx2.setDimension(x2.dimension());

    Dimension d1 = x1.dimension();
    Dimension d2 = x2.dimension();

    DEEPNET_ASSERT(d1.batch() == d2.batch());
    DEEPNET_ASSERT(d1.channel() == d2.channel());

    _y.setDimension(d1.batch(), d1.channel(), d1.height() + d2.height(),
                    d1.width());
}

LayerCat::~LayerCat() {}

void LayerCat::forward(const TensorGpu &x1, const TensorGpu &x2) {
    DEEPNET_TRACER;

    Layer::forward(x1);

    DEEPNET_ASSERT(x2.dimension() == _dx2.dimension());
    _px2 = &x2;

    gpu_cat_forward(_y.data(), x1.data(), x2.data(),        //
                    x1.batch(), x1.channel(), 
                    x1.height(), x2.height(), x1.width());
}

void LayerCat::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    Layer::backward(dy);

    gpu_cat_backward(dy.data(), _dx.data(), _dx2.data(),        //
                    _dx.batch(), _dx.channel(),
                    _dx.height(), _dx2.height(), _dx.width());
}

inline TensorGpu &LayerCat::dx2(void) { return _dx2; }

void LayerCat::print(tool::TablePrinter &printer, //
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
