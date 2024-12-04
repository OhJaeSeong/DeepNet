/// Copyright (c)2021 Electronics and Telecommunications Research
/// Institute(ETRI)

#pragma once

#include "deepnet/Weight.hpp"
#include "deepnet/Workspace.hpp"
#include "deepnet/layer/LayerBatchNorm.hpp"
#include "deepnet/layer/LayerFullyConnected.hpp"

namespace deepnet {
namespace layer {

/// FC와 배치 정규화를 함께 수행하는 레이어.
class LayerFCNorm : public LayerWeighted {
  protected:
    /// FC 레이어의 출력 채널 수.
    int _out_channel;
    /// 배치 정규화 레이어에 전달할 엡실론 값.
    double _epsilon;

  public:
    /// 컨볼루션 레이어.
    LayerFullyConnected *fc;
    /// 배치 정규화 레이어.
    LayerBatchNorm *norm;

  public:
    /// 생성자.
    LayerFCNorm(const TensorGpu &x, int out_channel, double epsilon = 0.0)
        : LayerWeighted(x), _out_channel(out_channel), _epsilon(epsilon), //
          fc(new LayerFullyConnected(x, out_channel, false)),
          norm(new LayerBatchNorm(fc->y(), epsilon)) {}

    /// 소멸자.
    virtual ~LayerFCNorm();

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "FCNorm"; }

    /// 학습 상태 여부를 설정한다.
    virtual void train() override;

    /// 학습 상태 여부를 설정한다.
    virtual void eval() override;

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

    /// 레이어 출력 정보를 콘솔에 출력한다.
    virtual void debug(tool::TablePrinter &printer, int depth = 0,
                       int index = 1) override;

    /// 가중치 정보를 갱신한다.
    virtual void
    update(float learning_rate,
           Weight::UpdateMethod method = Weight::UpdateMethod::SGD) override;

    /// 가중치 값의 체크섬을 계산한다.
    virtual unsigned char checksum(void) const override;
};

} // namespace layer
} // namespace deepnet
