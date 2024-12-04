/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerGroupNorm.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>

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
void gpu_groupnorm_forward(float *y, float *x, float *mean, float *var, float *weight, float *bias, int group, double epsilon,
                         int batch, int channel, int height, int width);

namespace deepnet {
namespace layer {

/// 생성자.
LayerGroupNorm::LayerGroupNorm(       //
    const TensorGpu &x, const TensorGpu &mean, const TensorGpu &var, int group, double epsilon)
    : LayerWeighted(x), _group(group), _epsilon(epsilon) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!x.isEmpty());

    // 가중치
    w.setDimension(1, x.channel(), 1, 1);
    b.setDimension(1, x.channel(), 1, 1);

    _y.setDimension(x.batch(), x.channel(), x.height(), x.width());
    DEEPNET_ASSERT(!_y.isEmpty());
}

void LayerGroupNorm::train(void) {
    LayerWeighted::train();

    dw.setDimension(w.dimension());
    db.setDimension(b.dimension());
}

void LayerGroupNorm::eval(void) {
    LayerWeighted::eval();

    dw.setDimension(0, 0, 0, 0);
    db.setDimension(0, 0, 0, 0);
}

void LayerGroupNorm::randomizeWeight(Weight::InitMethod method) {
    Weight::initializeWeight(w, method);
    Weight::initializeWeight(b, method);
}

void LayerGroupNorm::readWeight(FILE *file, Weight::Format format) {
    DEEPNET_TRACER;
    
    try {
        Weight::readWeight(file, w, format);
        Weight::readWeight(file, b, format);
    } catch (const std::exception &e) {
        DEEPNET_LOG("type = " << type());
        throw e;
    }
}

void LayerGroupNorm::writeWeight(FILE *file, Weight::Format format) const {
    DEEPNET_TRACER;

    try {
        Weight::writeWeight(file, w, format);
        Weight::writeWeight(file, b, format);
    } catch (const std::exception &e) {
        DEEPNET_LOG("type = " << type());
        throw e;
    }
}

void LayerGroupNorm::forward(const TensorGpu &x, const TensorGpu &mean, const TensorGpu &var) {
    DEEPNET_TRACER;

    try {
        Layer::forward(x);
        
        gpu_groupnorm_forward(_y.data(), x.data(), mean.data(), var.data(), w.data(), b.data(), _group, _epsilon,
                       x.batch(), x.channel(), x.height(), x.width());

    } catch (std::exception &e) {
        DEEPNET_LOG("Error = " << e.what());
        DEEPNET_LOG("x=" << (std::string)x.dimension()           //
                      << ", w=" << (std::string)w.dimension() //
                      << ", y=" << (std::string)_y.dimension());

        throw e;
    }
}

void LayerGroupNorm::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT_MSG(false, "Not implemented!");
}

void LayerGroupNorm::print(tool::TablePrinter &printer, //
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

void LayerGroupNorm::printWeight(tool::TablePrinter &printer, int depth,
                                int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());

    TensorCpu w_cpu(w);
    TensorCpu bias(b);

    float w_min = 0.0f, w_max = 0.0f;
    std::tie(w_min, w_max) = w_cpu.getMinMax();

    float bias_min = 0.0f, bias_max = 0.0f;
    std::tie(bias_min, bias_max) = bias.getMinMax();

    printer.addRow({std::string(depth, '-') + std::to_string(index),
                    std::string(type()), //
                    (std::string)output.dimension(),
                    std::string("w=") + SS(w_min) + "~" + SS(w_max) +  //
                        ", b=" + SS(bias_min) + "~" + SS(bias_max)});
}

} // namespace layer
} // namespace deepnet
