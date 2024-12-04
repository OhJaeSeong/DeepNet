/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerCalculate.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>

#define USE_CUDNN

// For CUDNN API,
// See https://docs.nvidia.com/deeplearning/cudnn/api/index.html

/// @param y 출력 텐서.
/// @param x 입력 텐서.
/// @param isadd 덧셈곰셈.
/// @param batch 배치 크기.
/// @param out_channel 출력 텐서의 채널 수.
/// @param y_height 출력 텐서의 높이.
/// @param y_width 출력 텐서의 폭.
/// @param in_channel 입력 텐서의 채널 수.
/// @param x_height 입력 텐서의 크기.
/// @param x_width 입력 텐서의 폭.
void gpu_calculate_forward(float *y, float *x, double number, int type, long n);

namespace deepnet {
namespace layer {

/// 생성자.
LayerCalculate::LayerCalculate(       //
    const TensorGpu &x, Workspace &workspace, double number, int type)
    : Layer(x), _workspace(&workspace), _number(number), _type(type) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!x.isEmpty());

    _y.setDimension(x.batch(), x.channel(), x.height(), x.width());
    DEEPNET_ASSERT(!_y.isEmpty());

    // 전방향 전파를 위한 작업공간의 크기를 계산한다.
    size_t required_workspace_size = 0;

    // 작업 공간의 크기가 부족하면, 확장한다.
    _workspace->enlarge(required_workspace_size);
}

void LayerCalculate::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    try {
        Layer::forward(x);
        long n = x.batch() * x.channel() * x.height() * x.width();
        gpu_calculate_forward(_y.data(), x.data(), _number, _type, n);

    } catch (std::exception &e) {
        DEEPNET_LOG("Error = " << e.what());
        DEEPNET_LOG("x=" << (std::string)x.dimension()           //
                      << ", y=" << (std::string)_y.dimension());

        throw e;
    }
}

void LayerCalculate::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT_MSG(false, "Not implemented!");
}

void LayerCalculate::print(tool::TablePrinter &printer, //
                               int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());

    printer.addRow({std::string(depth, '-') + std::to_string(index),        //
                    std::string(type()),                                    //
                    (std::string)output.dimension()});
}

} // namespace layer
} // namespace deepnet
