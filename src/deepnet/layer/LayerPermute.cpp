/// Copyright (c)2021 Electronics and Telecommunications Research
/// Institute(ETRI)

#include "deepnet/layer/LayerPermute.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>

void gpu_permute_forward(float *y, float *x,                                 //
                         int input_0, int input_1, int input_2, int input_3, //
                         int output_0, int output_1, int output_2,
                         int output_3, //
                         int order_0, int order_1, int order_2, int order_3);

namespace deepnet {
namespace layer {

LayerPermute::LayerPermute(const TensorGpu &x, int order_0, int order_1,
                           int order_2, int order_3)
    : Layer(x), _order_0(order_0), _order_1(order_1), _order_2(order_2),
      _order_3(order_3) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!x.isEmpty());
    DEEPNET_ASSERT(0 <= _order_0 && _order_0 <= 3);
    DEEPNET_ASSERT(0 <= _order_1 && _order_1 <= 3);
    DEEPNET_ASSERT(0 <= _order_2 && _order_2 <= 3);
    DEEPNET_ASSERT(0 <= _order_3 && _order_3 <= 3);
    DEEPNET_ASSERT(_order_0 + _order_1 + _order_2 + _order_3 == 6);

    // 입력 차원.
    _input_0 = x.batch();
    _input_1 = x.channel();
    _input_2 = x.height();
    _input_3 = x.width();

    int dim[4] = {_input_0, _input_1, _input_2, _input_3};

    // 출력 차원.
    _output_0 = dim[order_0];
    _output_1 = dim[order_1];
    _output_2 = dim[order_2];
    _output_3 = dim[order_3];

    // 출력 텐서.
    _y.setDimension(_output_0, _output_1, _output_2, _output_3);
}

void LayerPermute::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);

    gpu_permute_forward(_y.data(), x.data(),                        //
                        _input_0, _input_1, _input_2, _input_3,     //
                        _output_0, _output_1, _output_2, _output_3, //
                        _order_0, _order_1, _order_2, _order_3);
}

void LayerPermute::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT_MSG(false, "Not implemented!");
}

void LayerPermute::print(tool::TablePrinter &printer, //
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
