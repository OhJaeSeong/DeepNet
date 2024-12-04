/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// 풀링 레이어.

/// 평균 풀링과 최대값 풀링을 지원한다.
class LayerPermute : public Layer {
    int _input_0;
    int _input_1;
    int _input_2;
    int _input_3;
    int _output_0;
    int _output_1;
    int _output_2;
    int _output_3;
    int _order_0;
    int _order_1;
    int _order_2;
    int _order_3;

  public:
    /// 생성자.
    /// @param x 입력 텐서.
    /// @param order_0 1차원의 permutation.
    /// @param order_1 2차원의 permutation.
    /// @param order_2 3차원의 permutation.
    /// @param order_3 4차원의 permutation.
    LayerPermute(const TensorGpu &x, int order_0, int order_1, int order_2, int order_3);

    /// 소멸자.
    inline ~LayerPermute() {};

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Permute"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter& printer, int depth = 0, int index = 1) const override;
};

} // namespace layer
} // namespace deepnet