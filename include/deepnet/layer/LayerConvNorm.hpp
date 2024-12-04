/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Weight.hpp"
#include "deepnet/Workspace.hpp"
#include "deepnet/layer/LayerBatchNorm.hpp"
#include "deepnet/layer/LayerConvolutional.hpp"

namespace deepnet {
namespace layer {

/// 컨볼루션과 배치 정규화를 함께 수행하는 레이어.
class LayerConvNorm : public LayerWeighted {
  protected:
    /// 컨볼루션 레이어의 스트라이드 값.
    int _stride;
    /// 컨볼루션 레이어의 패딩 값.
    int _padding;
    /// 배치 정규화 레이어에 전달할 엡실론 값.
    double _epsilon;

  public:
    /// 컨볼루션 레이어.
    LayerConvolutional *conv;
    /// 배치 정규화 레이어.
    LayerBatchNorm *norm;

  public:
    /// 생성자.
    LayerConvNorm(const TensorGpu &x, Workspace &workspace, //
                  int out_channel, int height, int width,   //
                  int stride = 1, int padding = 0, double epsilon = 0.0)
        : _stride(stride), _padding(padding), _epsilon(epsilon),  //
          LayerWeighted(x),
          conv(new LayerConvolutional(x, workspace,               //
                                      out_channel, height, width, //
                                      stride, padding, false)),
          norm(new LayerBatchNorm(conv->y(), epsilon)) {}

    /// 소멸자.
    virtual ~LayerConvNorm();

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "ConvNorm"; }

    /// 학습 상태 여부를 설정한다.
    virtual void train(void) override;

    /// 학습 상태 여부를 설정한다.
    virtual void eval(void) override;

    /// 가중치 정보를 읽는다.
    virtual void readWeight(FILE *file, Weight::Format format) override;

    /// 가중치 정보를 저장한다.
    virtual void writeWeight(FILE *file, Weight::Format format) const override;

    /// 가중치 정보를 초기화한다.
    virtual void randomizeWeight(Weight::InitMethod method) override;

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 출력값을 반환한다.
    virtual const TensorGpu &y(void) const override;

    /// 델타 x 값을 반환한다.
    virtual const TensorGpu &dx(void) const override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter &printer, int depth = 0,
                       int index = 1) const override;

    /// 네트워크의 가중치 정보를 출력한다.
    virtual void printWeight(tool::TablePrinter &printer, //
                             int depth = 0, int index = 1) const override;

    /// 레이어 출력 정보를 콘솔에 출력한다.
    virtual void debug(tool::TablePrinter &printer, int depth = 0,
                       int index = 1) override;

    /// stride 값을 반환한다.
    inline int stride(void) { return _stride; }

    /// padding 값을 반환한다.
    inline int padding(void) { return _padding; }

    /// 가중치 정보를 갱신한다.
    virtual void
    update(float learning_rate,
           Weight::UpdateMethod method = Weight::UpdateMethod::SGD) override;

    /// 가중치 값의 체크섬을 계산한다.
    virtual unsigned char checksum(void) const override;
};

} // namespace layer
} // namespace deepnet
