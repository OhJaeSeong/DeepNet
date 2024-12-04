/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Weight.hpp"
#include "deepnet/Workspace.hpp"
#include "deepnet/layer/LayerWeighted.hpp"

namespace deepnet {
namespace layer {

/// 컨볼루션을 수행하는 레이어.
class LayerConvolutional : public LayerWeighted {
    int _out_channel;
    int _height;
    int _width;
    int _stride;
    int _padding;
    bool _bias;
    int _group;

    // 전방향 전파에 사용할 방법.
    cudnnConvolutionFwdAlgoPerf_t _fwd_perf_results;

    // // 역방향 전파 시 필터 갱신에 사용할 방법.
    cudnnConvolutionBwdFilterAlgoPerf_t _bwd_perf_results;

    // // 역방향 전파 시 입력값 갱신에 사용할 방법.
    cudnnConvolutionBwdDataAlgoPerf_t _bwd_data_perf_results;

    /// 작업 공간의 포인터.
    Workspace *_workspace;

  public:
    /// 생성자.
    LayerConvolutional(const TensorGpu &x, Workspace &workspace, //
                       int out_channel, int height, int width,   //
                       int stride = 1, int padding = 0, bool bias = true, int group = 1);

    virtual const char *type(void) const override { return "Convolutional"; }

    /// 학습 상태 여부를 설정한다.
    virtual void train(void) override;

    /// 학습 상태 여부를 설정한다.
    virtual void eval(void) override;

    /// 가중치 정보를 초기화한다.
    virtual void randomizeWeight(Weight::InitMethod method) override;

    /// 가중치 정보를 읽는다.
    virtual void readWeight(FILE *file, Weight::Format format) override;

    /// 가중치 정보를 저장한다.
    virtual void writeWeight(FILE *file, Weight::Format format) const override;

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter &printer, int depth = 0,
                       int index = 1) const override;

    /// 네트워크의 가중치 정보를 출력한다.
    virtual void printWeight(tool::TablePrinter &printer, //
                             int depth = 0, int index = 1) const override;

    /// 출력 채널의 수를 반환한다.
    inline int out_channel(void) { return _out_channel; }

    /// 높이 값을 반환한다.
    inline int height(void) { return _height; }

    /// 폭 값을 반환한다.
    inline int width(void) { return _width; }

    /// stride 값을 반환한다.
    inline int stride(void) { return _stride; }

    /// padding 값을 반환한다.
    inline int padding(void) { return _padding; }

    /// bias 여부를 반환한다.
    inline bool bias(void) { return _bias; }

    /// 가중치 값의 체크섬을 계산한다.
    virtual unsigned char checksum(void) const override;
};

} // namespace layer
} // namespace deepnet
