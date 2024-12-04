/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once
#include <functional>

#include "deepnet/Network.hpp"
#include "deepnet/Workspace.hpp"
#include "deepnet/layer/LayerConvNormLeaky.hpp"
#include "deepnet/layer/LayerMerge.hpp"
#include "deepnet/layer/LayerSequential.hpp"
#include "deepnet/layer/LayerSplit.hpp"

namespace deepnet {
namespace layer {

/// Darknet-53을 구성하는 레이어.
class LayerDarknet53A : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;

    LayerSplit *split;
    LayerConvNormLeaky *conv1, *conv2;
    LayerMerge *merge;

  public:
    /// 생성자.
    LayerDarknet53A(const TensorGpu &x, Workspace &workspace, //
                    int filter1, int filter2);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Darknet53A"; };

    /// 전방향 전파.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    virtual void backward(const TensorGpu &dy) override;
};

/// Darknet-53을 구성하는 레이어.
class LayerDarknet53B : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;

  public:
    /// 생성자.
    LayerDarknet53B(const TensorGpu &x, Workspace &workspace, int filter);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Darknet53B"; };
};

/// ModelDarknet53과 ModelYolov3에서 사용하는 특징 추출 레이어.

/// See "YOLOv3: An Incremental Improvement".
class LayerDarknet53 : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;
    int cb = 0;
  public:
    /// 36번째 레이어.
    LayerSplit *split36;
    /// 61번째 레이어.
    LayerSplit *split61;

    /// 생성자.
    /// @param x 입력 텐서.
    /// @param workspace 컨볼루션 레이어에서 사용하는 작업 공간.
    /// @param isDetector 분류기로 사용할 때와 객체 탐지기로 사용할 때의
    /// 네트워크 구조가 다르다. 탐지기로 사용하면 true.
    LayerDarknet53(const TensorGpu &x, Workspace &workspace, //
                   bool isDetector = false, std::function<void(int)> callback = nullptr);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Darknet53"; };
};

} // namespace layer
} // namespace deepnet
