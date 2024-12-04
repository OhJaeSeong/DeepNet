/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerConvTranspose.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>

#define USE_CUDNN

// For CUDNN API,
// See https://docs.nvidia.com/deeplearning/cudnn/api/index.html

/// @param y 출력 텐서.
/// @param x 입력 텐서.
/// @param k 커널 텐서.
/// @param batch 배치 크기.
/// @param out_channel 출력 텐서의 채널 수.
/// @param y_height 출력 텐서의 높이.
/// @param y_width 출력 텐서의 폭.
/// @param in_channel 입력 텐서의 채널 수.
/// @param x_height 입력 텐서의 크기.
/// @param x_width 입력 텐서의 폭.
/// @param k_height 커널 텐서의 높이.
/// @param k_width 커널 텐서의 폭.
/// @param stride 커널의 이동 속도.
/// @param padding 입력 텐서의 확장 크기.
void gpu_convtranspose_forward(                 //
    float *y, float *x, float *w, float *b,               //
    int batch,                                  //
    int out_channel, int y_height, int y_width, //
    int in_channel, int x_height, int x_width,  //
    int k_height, int k_width,                  //
    int stride, int padding);

namespace deepnet {
namespace layer {

/// 생성자.
LayerConvTranspose::LayerConvTranspose(       //
    const TensorGpu &x, Workspace &workspace, //
    int out_channel, int height, int width,   //
    int stride, int padding, bool bias, int group)
    : LayerWeighted(x), _workspace(&workspace),                  //
      _out_channel(out_channel), _height(height), _width(width), //
      _stride(stride), _padding(padding), _bias(bias), _group(group) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!x.isEmpty());
    DEEPNET_ASSERT(_out_channel > 0);
    DEEPNET_ASSERT(_height > 0);
    DEEPNET_ASSERT(_width > 0);
    DEEPNET_ASSERT(_stride > 0);
    DEEPNET_ASSERT(_padding >= 0);
    DEEPNET_ASSERT(x.height() >= _height);
    DEEPNET_ASSERT(x.width() >= _width);

    int in_channel = x.channel();

    // 가중치 w.
    w.setDimension(_out_channel, int(in_channel/_group), _height, _width);

    // 출력값 y의 차원을 설정한다.
    int out_n = x.batch();
    int out_c = _out_channel;
    int out_h = x.height(); // - (_stride - _height);
    int out_w = x.width(); // - (_stride - _width);
    _y.setDimension(out_n, out_c, out_h, out_w);
    DEEPNET_ASSERT(!_y.isEmpty());

    // 바이어스 항.
    if (_bias)
        b.setDimension(1, out_c, 1, 1);


    // 전방향 전파를 위한 작업공간의 크기를 계산한다.
    size_t required_workspace_size = 0;

    // 작업 공간의 크기가 부족하면, 확장한다.
    _workspace->enlarge(required_workspace_size);
}

void LayerConvTranspose::train(void) {
    LayerWeighted::train();

    dw.setDimension(w.dimension());

    // 바이어스 항.
    if (_bias)
        db.setDimension(1, _y.dimension().channel(), 1, 1);

    DEEPNET_ASSERT(_px);
    int algorithm_count = 0;
    size_t required_workspace_size = 0;

    DEEPNET_ASSERT(algorithm_count == 1);

    // 작업 공간의 크기가 부족하면, 확장한다.
    _workspace->enlarge(required_workspace_size);

    DEEPNET_ASSERT(algorithm_count == 1);

    // 작업 공간의 크기가 부족하면, 확장한다.
    _workspace->enlarge(required_workspace_size);
}

void LayerConvTranspose::eval(void) {
    LayerWeighted::eval();

    dw.setDimension(0, 0, 0, 0);
    db.setDimension(0, 0, 0, 0);
}

void LayerConvTranspose::randomizeWeight(Weight::InitMethod method) {
    Weight::initializeWeight(w, method);

    if (_bias)
        Weight::initializeWeight(b, method);
}

void LayerConvTranspose::readWeight(FILE *file, Weight::Format format) {
    DEEPNET_TRACER;

    try {
        if (format == Weight::Format::Darknet) {
            if (_bias)
                Weight::readWeight(file, b, format);
            Weight::readWeight(file, w, format);
        } else {
            Weight::readWeight(file, w, format);
            if (_bias)
                Weight::readWeight(file, b, format);
        }
    } catch (const std::exception &e) {
        DEEPNET_LOG("type = " << type());
        throw e;
    }
}

void LayerConvTranspose::writeWeight(FILE *file, Weight::Format format) const {
    DEEPNET_TRACER;

    try {
        if (format == Weight::Format::Darknet) {
            if (_bias)
                Weight::writeWeight(file, b, format);
            Weight::writeWeight(file, w, format);
        } else {
            Weight::writeWeight(file, w, format);
            if (_bias)
                Weight::writeWeight(file, b, format);
        }
    } catch (const std::exception &e) {
        DEEPNET_LOG("type = " << type());
        throw e;
    }
}

void LayerConvTranspose::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    try {
        Layer::forward(x);
        if(_bias){
            gpu_convtranspose_forward(_y.data(), x.data(), w.data(), b.data(), x.batch(), 
                                _y.height(), _y.width(), _y.channel(), // _out_channel 
                                x.height(), x.width(), x.channel(),
                                _height, _width, _stride, _padding);
        }else{
            gpu_convtranspose_forward(_y.data(), x.data(), w.data(), NULL, x.batch(), 
                                _y.height(), _y.width(), _y.channel(), // _out_channel 
                                x.height(), x.width(), x.channel(),
                                _height, _width, _stride, _padding);
        }
        

    } catch (std::exception &e) {
        DEEPNET_LOG("Error = " << e.what());
        DEEPNET_LOG("x=" << (std::string)x.dimension()           //
                      << ", w=" << (std::string)w.dimension() //
                      << ", y=" << (std::string)_y.dimension());

        throw e;
    }
}

void LayerConvTranspose::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

}

void LayerConvTranspose::print(tool::TablePrinter &printer, //
                               int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());
    auto &filter = w;

    printer.addRow({std::string(depth, '-') + std::to_string(index),        //
                    std::string(type()),                                    //
                    (std::string)output.dimension(),                        //
                    (std::string)filter.dimension(),                        //
                    std::string("stride=") + std::to_string(_stride) + ", " //
                        + "padding=" + std::to_string(_padding) + ", "      //
                        + "bias=" + (_bias ? "true" : "false")});
}

#define SS(x) std::to_string(x)

void LayerConvTranspose::printWeight(tool::TablePrinter &printer, int depth,
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

unsigned char LayerConvTranspose::checksum(void) const {
    auto checksum_w = TensorCpu(w).checksum();

    if (!_bias)
        return checksum_w;

    auto checksum_b = TensorCpu(b).checksum();

    return ~(~checksum_w + ~checksum_b);
}

} // namespace layer
} // namespace deepnet
