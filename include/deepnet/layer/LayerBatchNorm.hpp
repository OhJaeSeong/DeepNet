/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Weight.hpp"
#include "deepnet/layer/LayerWeighted.hpp"

namespace deepnet {
namespace layer {

/// 배치 정규화 레이어.
class LayerBatchNorm : public LayerWeighted {
    double _epsilon;
    double _n;

    cudnnTensorDescriptor_t _bnScaleBiasMeanVarDesc;

    TensorGpu _resultSaveMean;
    TensorGpu _resultSaveInvVariance;

  public:
    /// 평균 값.
    TensorGpu mean;
    /// 분산 값.
    TensorGpu variance;

  public:
    /// 생성자.
    LayerBatchNorm(const TensorGpu &x, double epsilon = 0.0f);

    /// 소멸자.
    virtual ~LayerBatchNorm();

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "BatchNorm"; }

    /// 학습 상태 여부를 설정한다.
    virtual void train(void) override;

    /// 학습 상태 여부를 설정한다.
    virtual void eval(void) override;

    /// 가중치 정보를 읽는다.
    /// 파일에서 BCHW의 순서로 읽고, 텐서에 BHWC의 순서로 저장한다.
    virtual void readWeight(FILE *file, Weight::Format format) override;

    /// 가중치 정보를 저장한다.
    /// 텐서에서 BHWC의 순서로 읽고, 파일에 BCHW의 순서로 저장한다.
    virtual void writeWeight(FILE *file, Weight::Format format) const override;

    /// 가중치 정보를 초기화한다.
    virtual void randomizeWeight(Weight::InitMethod method) override;

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

    /// epsilon 값을 반환한다.
    inline double epsilon(void) { return _epsilon; }

    /// 가중치 값의 체크섬을 계산한다.
    virtual unsigned char checksum(void) const override;
};

} // namespace layer
} // namespace deepnet
