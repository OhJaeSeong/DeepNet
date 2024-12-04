/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerImplicit.hpp"
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
void gpu_implicit_forward(float *y, float *x, float *weight,
                         int batch, int channel, int height, int width, bool isadd);

namespace deepnet {
namespace layer {

/// 생성자.
LayerImplicit::LayerImplicit(       //
    const TensorGpu &x, Workspace &workspace, bool isadd)
    : LayerWeighted(x), _workspace(&workspace), _isadd(isadd) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!x.isEmpty());

    // 가중치 w.
    w.setDimension(1, x.channel(), 1, 1);

    _y.setDimension(x.batch(), x.channel(), x.height(), x.width());
    DEEPNET_ASSERT(!_y.isEmpty());

    // 전방향 전파를 위한 작업공간의 크기를 계산한다.
    size_t required_workspace_size = 0;

    // 작업 공간의 크기가 부족하면, 확장한다.
    _workspace->enlarge(required_workspace_size);
}

void LayerImplicit::train(void) {
    LayerWeighted::train();

    dw.setDimension(w.dimension());
}

void LayerImplicit::eval(void) {
    LayerWeighted::eval();

    dw.setDimension(0, 0, 0, 0);
}

void LayerImplicit::randomizeWeight(Weight::InitMethod method) {
    Weight::initializeWeight(w, method);
}

void LayerImplicit::readWeight(FILE *file, Weight::Format format) {
    DEEPNET_TRACER;
    
    try {
        Weight::readWeight(file, w, format);
    } catch (const std::exception &e) {
        DEEPNET_LOG("type = " << type());
        throw e;
    }
}

void LayerImplicit::writeWeight(FILE *file, Weight::Format format) const {
    DEEPNET_TRACER;

    try {
        Weight::writeWeight(file, w, format);
    } catch (const std::exception &e) {
        DEEPNET_LOG("type = " << type());
        throw e;
    }
}

void LayerImplicit::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    try {
        Layer::forward(x);

        gpu_implicit_forward(_y.data(), x.data(), w.data(),                   //
                        x.batch(), x.channel(), x.height(), x.width(), _isadd);

    } catch (std::exception &e) {
        DEEPNET_LOG("Error = " << e.what());
        DEEPNET_LOG("x=" << (std::string)x.dimension()           //
                      << ", w=" << (std::string)w.dimension() //
                      << ", y=" << (std::string)_y.dimension());

        throw e;
    }
}

void LayerImplicit::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT_MSG(false, "Not implemented!");
}

void LayerImplicit::print(tool::TablePrinter &printer, //
                               int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());
    auto &filter = w;

    printer.addRow({std::string(depth, '-') + std::to_string(index),        //
                    std::string(type()),                                    //
                    (std::string)output.dimension(),                        //
                    (std::string)filter.dimension()});
}

#define SS(x) std::to_string(x)

void LayerImplicit::printWeight(tool::TablePrinter &printer, int depth,
                                int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());

    TensorCpu w_cpu(w);
    float w_min = 0.0f, w_max = 0.0f;
    std::tie(w_min, w_max) = w_cpu.getMinMax();

    printer.addRow({std::string(depth, '-') + std::to_string(index),
                    std::string(type()), //
                    (std::string)output.dimension(),
                    std::string("w=") + SS(w_min) + "~" + SS(w_max)});
}

} // namespace layer
} // namespace deepnet
