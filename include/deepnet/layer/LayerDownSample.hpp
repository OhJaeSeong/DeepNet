/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"
#include "deepnet/layer/LayerPooling.hpp"

namespace deepnet {
namespace layer {

/// 다운 샘플링을 수행하는 레이어.

/// DownSample을 AveragePooling으로 대체한다.
class LayerDownSample : public LayerPooling {
    int _sample;

  public:
    /// 생성자.
    LayerDownSample(const TensorGpu &x, int sample = 2);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "DownSample"; }

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter& printer, int depth = 0, int index = 1) const override;

    /// sample 값을 반환한다.
    inline int sample(void) { return _sample; }
};

} // namespace layer
} // namespace deepnet
