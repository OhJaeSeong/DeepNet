/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// 풀링 레이어.

/// 평균 풀링과 최대값 풀링을 지원한다.
class LayerPooling : public Layer {
    int _size_height;
    int _size_width;
    int _stride_height;
    int _stride_width;
    int _padding_height;
    int _padding_width;
    bool _max;

  protected:
    /// 풀링 연산 관련 정보를 저장한다.
    cudnnPoolingDescriptor_t _pool_desc;

  public:
    /// 생성자.
    inline LayerPooling(const TensorGpu &x, int size, int stride, int padding = 0,
                        bool max = true)
        : LayerPooling(x, size, size, stride, stride, padding, padding, max) {}

    /// 생성자.
    /// @param x 입력 텐서.
    /// @param size_height 풀링 높이. (0이면, x.height)
    /// @param size_width 풀링 폭. (0이면, x.width)
    /// @param stride_height 스트라이드 높이.
    /// @param stride_width 스트라이드 폭.
    /// @param padding_height 패딩 높이.
    /// @param padding_width 패딩 폭.
    /// @param max maxpooling 여부. (true이면 max pooling, false이면 average pooling)
    LayerPooling(const TensorGpu &x, int size_height, int size_width,
                 int stride_height, int stride_width, int padding_height,
                 int padding_width, bool max = true);

    /// 소멸자.
    ~LayerPooling();

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Pooling"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter& printer, int depth = 0, int index = 1) const override;
};

} // namespace layer
} // namespace deepnet