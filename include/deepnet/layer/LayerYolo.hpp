/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/layer/Layer.hpp"
#include <vector>

namespace deepnet {
namespace layer {

/// Yolo 모델에서 출력 값을 좌표값으로 변환하는 레이어.

/// 객체 신뢰값은 sigmoid 함수를 이용하여 0~1 범위의 값으로 변환한다.
class LayerYolo : public Layer {
    /// 원본 영상의 높이.
    int _image_height;

    /// 원본 영상의 폭.
    int _image_width;

    /// 각 그리드의 후보의 수.
    int _candidates;

    /// 클래스 수.
    int _classes;

    /// 앵커.
    float *_anchors;

  public:
    /// 생성자.
    LayerYolo(const TensorGpu &x,                   //
              int image_height, int image_width, //
              int candidate, int classes,        //
              std::vector<std::pair<int, int>> &anchors);

    /// 소멸자.
    ~LayerYolo();

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Yolo"; }

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;
};

} // namespace layer
} // namespace deepnet
