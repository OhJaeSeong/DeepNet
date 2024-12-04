/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/LayerWeighted.hpp"

namespace deepnet {
namespace layer {

/// Fully-connected 레이어.
class LayerFullyConnected : public LayerWeighted {
    /// 출력 채널의 수.
    int _out_channel;

    /// 1 벡터(1, batch).
    TensorGpu ones;

  public:
    /// c = a * b.
    static void mul(TensorGpu &c, const TensorGpu &a, const TensorGpu &b,
                    bool transa = false, bool transb = false);

  public:
    /// 생성자.
    /// @param x 입력 텐서.
    /// @param out_channel  출력 채널의 수.
    /// @param bias 바이어스 사용 여부.
    LayerFullyConnected(const TensorGpu &x, int out_channel, bool bias = true);

    virtual const char *type(void) const override { return "FullyConnected"; }

    /// 학습 상태 여부를 설정한다.
    virtual void train(void) override;

    /// 학습 상태 여부를 설정한다.
    virtual void eval(void) override;

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter& printer, int depth = 0, int index = 1) const override;

    /// 출력 채널의 수를 반환한다.
    inline int outChannel(void) { return _out_channel; }
};

} // namespace layer
} // namespace deepnet
