/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Weight.hpp"
#include "deepnet/layer/LayerWeighted.hpp"

namespace deepnet {
namespace layer {

/// 정규화 레이어.
class LayerNormalization : public LayerWeighted {
  protected:
    /// 텐서 덧셈 연산 관련 정보를 저장한다.
    cudnnOpTensorDescriptor_t _op_desc_add;

    /// 텐서 곱셈 연산 관련 정보를 저장한다.
    cudnnOpTensorDescriptor_t _op_desc_mul;

  public:
    /// 생성자.
    LayerNormalization(const TensorGpu &x);

    /// 소멸자.
    virtual ~LayerNormalization();

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Normalization"; }

    /// 가중치 정보를 읽는다.
    virtual void readWeight(FILE *file, Weight::Format format) override;

    /// 가중치 정보를 저장한다.
    virtual void writeWeight(FILE *file, Weight::Format format) const override;

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter &printer, //
                       int depth = 0, int index = 1) const override;

    /// 네트워크의 가중치 정보를 출력한다.
    virtual void printWeight(tool::TablePrinter &printer, //
                             int depth = 0, int index = 1) const override;

    /// 가중치 값의 체크섬을 계산한다.
    virtual unsigned char checksum(void) const override;
};

} // namespace layer
} // namespace deepnet
