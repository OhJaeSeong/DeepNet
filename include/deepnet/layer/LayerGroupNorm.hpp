/// Copyright (c)2022 HanulSoft(HNS)

#pragma once

#include "deepnet/Weight.hpp"
#include "deepnet/Workspace.hpp"
#include "deepnet/layer/LayerWeighted.hpp"

namespace deepnet {
namespace layer {

/// 그룹 정규화(group norm)를 수행한다.
class LayerGroupNorm : public LayerWeighted {

    int _group;
    double _epsilon;

  public:
    /// 생성자.
    LayerGroupNorm(const TensorGpu &x, const TensorGpu &mean, const TensorGpu &var, int group, double epsilon);

    virtual const char *type(void) const override { return "GroupNorm"; }

    /// 학습 상태 여부를 설정한다.
    virtual void train(void) override;

    /// 학습 상태 여부를 설정한다.
    virtual void eval(void) override;

    /// 가중치 정보를 초기화한다.
    virtual void randomizeWeight(Weight::InitMethod method) override;

    /// 가중치 정보를 읽는다.
    virtual void readWeight(FILE *file, Weight::Format format) override;

    virtual void writeWeight(FILE *file, Weight::Format format) const override;

    virtual void forward(const TensorGpu &x) override {
        DEEPNET_LOG("Unused forward");
        DEEPNET_ASSERT(false);
    }

    void forward(const TensorGpu &x, const TensorGpu &mean, const TensorGpu &var);

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
