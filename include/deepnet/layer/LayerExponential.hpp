/// Copyright (c)2022 HanulSoft(HNS)

#pragma once

#include "deepnet/Workspace.hpp"
#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// group norm을 위해 텐서를 지정된 수만큼의 그룹별로 묶어주는 역할을 수행하는 연산
class LayerExponential : public Layer {
    int _group;

  public:
    /// 생성자.
    LayerExponential(const TensorGpu &x, int group);

    virtual const char *type(void) const override { return "Exponential"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter &printer, int depth = 0,
                       int index = 1) const override;

};

} // namespace layer
} // namespace deepnet
