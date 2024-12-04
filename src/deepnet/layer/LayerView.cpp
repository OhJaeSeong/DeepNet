/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerView.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>

void gpu_view_forward(float *y, float *x, int sum);

namespace deepnet {
namespace layer {

LayerView::LayerView(const TensorGpu &x, int order_0, int order_1,
                           int order_2, int order_3)
    : Layer(x), _order_0(order_0), _order_1(order_1), _order_2(order_2),
      _order_3(order_3) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!x.isEmpty());
    DEEPNET_ASSERT(_order_0 * _order_1 * _order_2 * _order_3 == x.batch() * x.channel() * x.height() * x.width());

    // 출력 텐서.
    _y.setDimension(order_0, order_1, order_2, order_3);
}

void LayerView::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);    
    gpu_view_forward(_y.data(), x.data(), _order_0 * _order_1 * _order_2 * _order_3);
}

void LayerView::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT_MSG(false, "Not implemented!");
}

void LayerView::print(tool::TablePrinter &printer, //
                         int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = y();
    std::stringstream option;

    option << "order=("        //
           << _order_0 << ", " //
           << _order_1 << ", " //
           << _order_2 << ", " //
           << _order_3 << ")";

    printer.addRow({std::string(depth, '-') + std::to_string(index), //
                    std::string(type()),                             //
                    (std::string)output.dimension(),                 //
                    "",                                              //
                    option.str()});
}

} // namespace layer
} // namespace deepnet
