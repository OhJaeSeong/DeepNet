/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerConvolutional.hpp"
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
void gpu_convolutional_forward(                 //
    float *y, float *x, float *k,               //
    int batch,                                  //
    int out_channel, int y_height, int y_width, //
    int in_channel, int x_height, int x_width,  //
    int k_height, int k_width,                  //
    int stride, int padding);

namespace deepnet {
namespace layer {

/// 생성자.
LayerConvolutional::LayerConvolutional(       //
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

    SAFE_CUDNN(cudnnCreateFilterDescriptor(&_w_desc));
    SAFE_CUDNN(cudnnSetFilter4dDescriptor(_w_desc, CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW, _out_channel,
                                          int(in_channel/_group), _height, _width));

    // 컨볼루션 정보.
    SAFE_CUDNN(cudnnCreateConvolutionDescriptor(&_conv_desc));
    SAFE_CUDNN(cudnnSetConvolution2dDescriptor(
        _conv_desc, _padding, _padding, _stride, _stride, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    DEEPNET_ASSERT(_w_desc);
    DEEPNET_ASSERT(_conv_desc);

    SAFE_CUDNN(cudnnSetConvolutionGroupCount(
        _conv_desc, _group));

    // 입력값 x와 가중치 w를 이용하여 출력값 y의 차원을 계산한다.
    int out_n, out_c, out_h, out_w;
    SAFE_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        _conv_desc, x.descriptor(), _w_desc, &out_n, &out_c, &out_h, &out_w));

    // 출력값 y의 차원을 설정한다.
    _y.setDimension(out_n, out_c, out_h, out_w);
    DEEPNET_ASSERT(!_y.isEmpty());

    // 바이어스 항.
    if (_bias)
        b.setDimension(1, out_c, 1, 1);

    // 최적의 전방향 컨볼루션 알고리즘을 선택한다.
    int algorithm_count = 0;

    SAFE_CUDNN(cudnnFindConvolutionForwardAlgorithm( //
        cudnn_handle,                                //
        x.descriptor(),                              //
        _w_desc,                                     //
        _conv_desc,                                  //
        _y.descriptor(),                             //
        1,                                           //
        &algorithm_count,                            //
        &_fwd_perf_results));

    DEEPNET_ASSERT(algorithm_count == 1);

    // 전방향 전파를 위한 작업공간의 크기를 계산한다.
    size_t required_workspace_size = 0;

    SAFE_CUDNN(cudnnGetConvolutionForwardWorkspaceSize( //
        cudnn_handle,                                   //
        x.descriptor(),                                 //
        _w_desc,                                        //
        _conv_desc,                                     //
        _y.descriptor(),                                //
        _fwd_perf_results.algo,                         //
        &required_workspace_size));

    // 작업 공간의 크기가 부족하면, 확장한다.
    _workspace->enlarge(required_workspace_size);
}

void LayerConvolutional::train(void) {
    LayerWeighted::train();

    dw.setDimension(w.dimension());

    // 바이어스 항.
    if (_bias)
        db.setDimension(1, _y.dimension().channel(), 1, 1);

    DEEPNET_ASSERT(_px);
    int algorithm_count = 0;
    size_t required_workspace_size = 0;

    SAFE_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm( //
        cudnn_handle,                                       //
        _px->descriptor(),                                  //
        _y.descriptor(),                                    //
        _conv_desc,                                         //
        _w_desc,                                            //
        1,                                                  //
        &algorithm_count,                                   //
        &_bwd_perf_results));

    DEEPNET_ASSERT(algorithm_count == 1);

    SAFE_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize( //
        cudnn_handle,                                          //
        _px->descriptor(),                                     //
        _y.descriptor(),                                       //
        _conv_desc,                                            //
        _w_desc,                                               //
        _bwd_perf_results.algo,                                //
        &required_workspace_size));

    // 작업 공간의 크기가 부족하면, 확장한다.
    _workspace->enlarge(required_workspace_size);

    SAFE_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm( //
        cudnn_handle,                                     //
        _w_desc,                                          //
        _y.descriptor(),                                  //
        _conv_desc,                                       //
        _px->descriptor(),                                //
        1,                                                //
        &algorithm_count,                                 //
        &_bwd_data_perf_results));

    DEEPNET_ASSERT(algorithm_count == 1);

    SAFE_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize( //
        cudnn_handle,                                        //
        _w_desc,                                             //
        _y.descriptor(),                                     //
        _conv_desc,                                          //
        _px->descriptor(),                                   //
        _bwd_data_perf_results.algo,                         //
        &required_workspace_size));

    // 작업 공간의 크기가 부족하면, 확장한다.
    _workspace->enlarge(required_workspace_size);
}

void LayerConvolutional::eval(void) {
    LayerWeighted::eval();

    dw.setDimension(0, 0, 0, 0);
    db.setDimension(0, 0, 0, 0);
}

void LayerConvolutional::randomizeWeight(Weight::InitMethod method) {
    Weight::initializeWeight(w, method);

    if (_bias)
        Weight::initializeWeight(b, method);
}

void LayerConvolutional::readWeight(FILE *file, Weight::Format format) {
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

void LayerConvolutional::writeWeight(FILE *file, Weight::Format format) const {
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

void LayerConvolutional::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    try {
        Layer::forward(x);

        float alpha = 1.0f;
        float beta = 0.0f;

        // 순방향 전파를 실행한다.
        // ResNet-18 모델 실행 시 cudnnConvolutionForward() 함수는
        // GTX 1080에서 총 22ms 중 20ms의 시간을 소요하고,
        // RTX 2060에서 총 12ms 중 10ms의 시간을 소요한다.
        SAFE_CUDNN(cudnnConvolutionForward( //
            cudnn_handle,                   //
            &alpha,                         //
            x.descriptor(),                 //
            x.data(),                       //
            _w_desc,                        //
            w.data(),                       //
            _conv_desc,                     //
            _fwd_perf_results.algo,         //
            _workspace->data(),             //
            _workspace->size(),             //
            &beta,                          //
            _y.descriptor(),                //
            _y.data()));

        // 바이어스 항을 추가한다.
        if (_bias)
            SAFE_CUDNN(cudnnAddTensor( //
                cudnn_handle,          //
                &alpha,                //
                b.descriptor(),        //
                b.data(),              //
                &alpha,                // 값을 누적한다.
                _y.descriptor(),       //
                _y.data()));

    } catch (std::exception &e) {
        DEEPNET_LOG("Error = " << e.what());
        DEEPNET_LOG("x=" << (std::string)x.dimension()           //
                      << ", w=" << (std::string)w.dimension() //
                      << ", y=" << (std::string)_y.dimension());

        throw e;
    }
}

void LayerConvolutional::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    try {
        Layer::backward(dy);

        float alpha = 1.0f;
        float beta = 0.0f;

        // 가중치의 델타(dw)를 계산한다.
        SAFE_CUDNN(cudnnConvolutionBackwardFilter( //
            cudnn_handle,                          //
            &alpha,                                //
            _px->descriptor(),                     //
            _px->data(),                           //
            dy.descriptor(),                       //
            dy.data(),                             //
            _conv_desc,                            //
            _bwd_perf_results.algo,                //
            _workspace->data(),                    //
            _workspace->size(),                    //
            &beta,                                 //
            _w_desc,                               //
            dw.data()));

        // 바이어스 항의 델타(db)를 계산한다.
        if (_bias)
            SAFE_CUDNN(cudnnConvolutionBackwardBias( //
                cudnn_handle,                        //
                &alpha,                              //
                dy.descriptor(),                     //
                dy.data(),                           //
                &beta,                               //
                db.descriptor(),                     //
                db.data()));

        // 입력 벡터의 델타(dx)를 계산한다.
        SAFE_CUDNN(cudnnConvolutionBackwardData( //
            cudnn_handle,                        //
            &alpha,                              //
            _w_desc,                             //
            w.data(),                            //
            dy.descriptor(),                     //
            dy.data(),                           //
            _conv_desc,                          //
            _bwd_data_perf_results.algo,         //
            _workspace->data(),                  //
            _workspace->size(),                  //
            &beta,                               //
            _dx.descriptor(),                    //
            _dx.data()));
    } catch (std::exception &e) {
        DEEPNET_LOG("dx=" << (std::string)_dx.dimension() //
                       << ", w=" << (std::string)w.dimension()
                       << ", dy=" << (std::string)dy.dimension());

        throw e;
    }
}

void LayerConvolutional::print(tool::TablePrinter &printer, //
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

void LayerConvolutional::printWeight(tool::TablePrinter &printer, int depth,
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

unsigned char LayerConvolutional::checksum(void) const {
    auto checksum_w = TensorCpu(w).checksum();

    if (!_bias)
        return checksum_w;

    auto checksum_b = TensorCpu(b).checksum();

    return ~(~checksum_w + ~checksum_b);
}

} // namespace layer
} // namespace deepnet
