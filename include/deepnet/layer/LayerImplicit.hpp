/// Copyright (c)2022 HanulSoft(HNS)

#pragma once

#include "deepnet/Weight.hpp"
#include "deepnet/Workspace.hpp"
#include "deepnet/layer/LayerWeighted.hpp"

namespace deepnet {
namespace layer {

/// 사칙연산(덧셈, 곱셈)을 수행하는 레이어. calculate랑 다르게 고정된 숫자가 아닌 레이어 내 가중치를 활용한 연산을 한다.
class LayerImplicit : public LayerWeighted {
    /// 작업 공간의 포인터.
    Workspace *_workspace;
    bool _isadd;

  public:
    /// 생성자.
    LayerImplicit(const TensorGpu &x, Workspace &workspace, bool isadd = true);

    virtual const char *type(void) const override { return "Implicit"; }

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
};

} // namespace layer
} // namespace deepnet
