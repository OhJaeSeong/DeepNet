/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Network.hpp"
#include "deepnet/Workspace.hpp"
#include "deepnet/layer/LayerSequential.hpp"

namespace deepnet {
namespace layer {

/// ModelDarknet19와 ModelYolov2에서 사용하는 특징 추출 레이어.

/// See "YOLO9000: Better, Faster, Stronger".
class LayerDarknet19 : public LayerSequential {
    /// 작업 공간.
    Workspace *_workspace;

    inline void conv(const TensorGpu &x, int channel, int size);
    inline void pool(const TensorGpu &x);

  public:
    /// 생성자.
    LayerDarknet19(const TensorGpu &x, Workspace &workspace);

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Darknet19"; };
};

} // namespace layer
} // namespace deepnet
