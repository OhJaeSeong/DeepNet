/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerSeparate.hpp"
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
void gpu_separate_forward(float *x, float *y, int number, int dim, int count, int channel, int height, int width);

namespace deepnet {
namespace layer {

/// 생성자.
LayerSeparate::LayerSeparate(const TensorGpu &x, int dim, int range1, int range2) 
    // number 분할갯수, dum 분할하는 차원, count y값으로 반환할 값(몇번째 것을 반환?)
    : Layer(x), _dim(dim), _range1(range1), _range2(range2) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!x.isEmpty());

    if(dim == 1){
        _y.setDimension(x.batch(), int(range2 - range1), x.height(), x.width());
    }else if(dim == 2){
        _y.setDimension(x.batch(), x.channel(), int(range2 - range1), x.width());
    }else if(dim == 3){
        _y.setDimension(x.batch(), x.channel(), x.height(), int(range2 - range1));
    }
    
    DEEPNET_ASSERT(!_y.isEmpty());
}

void LayerSeparate::forward(const TensorGpu &x) { // , int number, int dim, int count
    DEEPNET_TRACER;
    try {
        Layer::forward(x);
        gpu_separate_forward(x.data(), _y.data(), _dim, _range1, _range2, x.channel(), x.height(), x.width());

    } catch (std::exception &e) {
        DEEPNET_LOG("Error = " << e.what());
        DEEPNET_LOG("x=" << (std::string)x.dimension()           //
                      << ", y=" << (std::string)_y.dimension());

        throw e;
    }
}

void LayerSeparate::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT_MSG(false, "Not implemented!");
}

void LayerSeparate::print(tool::TablePrinter &printer, //
                               int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());

    printer.addRow({std::string(depth, '-') + std::to_string(index),        //
                    std::string(type()),                                    //
                    (std::string)output.dimension()});
}

} // namespace layer
} // namespace deepnet
