/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerCalculateEach.hpp"
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
void gpu_calculateE_forward(float *x1, float *x2, float *y, int channel, int height, int width, int type, int duple);
// 0 : add, 1 : subtract, 2 : multiple, 3 : divide 

namespace deepnet {
namespace layer {

/// 생성자.
LayerCalculateEach::LayerCalculateEach(const TensorGpu &x1, const TensorGpu &x2, int type) 
    // number 분할갯수, dum 분할하는 차원, count y값으로 반환할 값(몇번째 것을 반환?)
    : Layer(x1), _type(type) {
    DEEPNET_TRACER;

    _dx2.setDimension(x1.dimension());

    Dimension d1 = x1.dimension();
    Dimension d2 = x2.dimension();

    DEEPNET_ASSERT(d1.batch() == d2.batch());
    // DEEPNET_ASSERT(d1.channel() == d2.channel());
    DEEPNET_ASSERT(d1.height() == d2.height());
    DEEPNET_ASSERT(d1.width() == d2.width());

    DEEPNET_ASSERT(!x1.isEmpty());
    DEEPNET_ASSERT(!x2.isEmpty());
    _y.setDimension(x1.batch(), x1.channel(), x1.height(), x1.width());
    
    DEEPNET_ASSERT(!_y.isEmpty());
}

void LayerCalculateEach::forward(const TensorGpu &x1, const TensorGpu &x2) {
    DEEPNET_TRACER;
    try {
        Layer::forward(x1);
        int duple = int(x1.channel() / x2.channel());
        gpu_calculateE_forward(x1.data(), x2.data(), _y.data(), x1.channel(), x1.height(), x1.width(), _type, duple);

    } catch (std::exception &e) {
        DEEPNET_LOG("Error = " << e.what());
        DEEPNET_LOG("x=" << (std::string)x1.dimension()           //
                      << ", y=" << (std::string)_y.dimension());

        throw e;
    }
}

void LayerCalculateEach::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT_MSG(false, "Not implemented!");
}

void LayerCalculateEach::print(tool::TablePrinter &printer, //
                               int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());

    printer.addRow({std::string(depth, '-') + std::to_string(index),        //
                    std::string(type()),                                    //
                    (std::string)output.dimension()});
}

} // namespace layer
} // namespace deepnet
