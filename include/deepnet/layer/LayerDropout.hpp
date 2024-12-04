/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// 드롭 아웃을 수행하는 레이어.
class LayerDropout : public Layer {
    float _dropout;
    cudnnDropoutDescriptor_t _descriptor;
    size_t _state_size;
    void *_states; // 드롭된 가중치의 위치 정보를 저장한다.
    size_t _space_size;
    void *_spaces; // 드롭된 가중치의 값 정보를 저장한다.

  public:
    /// 생성자.
    LayerDropout(const TensorGpu &x, float dropout = 0.5f);

    /// 소멸자.
    virtual ~LayerDropout();

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Dropout"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter& printer, int depth = 0, int index = 1) const override;
};

} // namespace layer
} // namespace deepnet
