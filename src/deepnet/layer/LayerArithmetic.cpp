/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerArithmetic.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>
#include <iostream>

#define USE_CUDNN

// For CUDNN API,
// See https://docs.nvidia.com/deeplearning/cudnn/api/index.html

/// @param y 출력 텐서.
/// @param x 입력 텐서.
/// @param batch 배치 크기.
/// @param out_channel 출력 텐서의 채널 수.
/// @param y_height 출력 텐서의 높이.
/// @param y_width 출력 텐서의 폭.
/// @param in_channel 입력 텐서의 채널 수.
/// @param x_height 입력 텐서의 크기.
/// @param x_width 입력 텐서의 폭.
void gpu_arithmetic_forward(float *x, float *y, int tensor_size, float number, int type); // 0 : add, 1 : subtract, 2 : multiple, 3: divide

namespace deepnet {
namespace layer {

/// 생성자.
LayerArithmetic::LayerArithmetic(const TensorGpu &x, float number, int type) 
    // number 분할갯수, dum 분할하는 차원, count y값으로 반환할 값(몇번째 것을 반환?)
    : Layer(x), _number(number), _type(type) {
    DEEPNET_TRACER;

    _dx2.setDimension(x.dimension());
    _y.setDimension(x.batch(), x.channel(), x.height(), x.width());
    
    DEEPNET_ASSERT(!_y.isEmpty());
}

void LayerArithmetic::forward(const TensorGpu &x) { // , int number, int dim, int count
    DEEPNET_TRACER;
    try {
        Layer::forward(x);
        gpu_arithmetic_forward(x.data(), _y.data(), x.channel() * x.height() * x.width(), _number, _type);

    } catch (std::exception &e) {
        DEEPNET_LOG("Error = " << e.what());
        DEEPNET_LOG("x=" << (std::string)x.dimension()           //
                      << ", y=" << (std::string)_y.dimension());

        throw e;
    }
}

void LayerArithmetic::print(tool::TablePrinter &printer, //
                               int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());

    printer.addRow({std::string(depth, '-') + std::to_string(index),        //
                    std::string(type()),                                    //
                    (std::string)output.dimension()});
}

} // namespace layer
} // namespace deepnet
