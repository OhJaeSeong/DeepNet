/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// 업 샘플링을 수행하는 레이어.
class LayerUpSample : public Layer {
    int _sample;

  protected:
    /// 풀링 연산 관련 정보를 저장한다.
    /// 업 샘플링의 역방향 전파는 풀링 연산으로 대체한다.
    cudnnPoolingDescriptor_t _pool_desc;

  public:
    /// 생성자.
    LayerUpSample(const TensorGpu &x, int sample);

    /// 소멸자.
    ~LayerUpSample();

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "UpSample"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter& printer, int depth = 0, int index = 1) const override;

    /// sample 값을 반환한다.
    inline int sample(void) { return _sample; }
};

} // namespace layer
} // namespace deepnet
