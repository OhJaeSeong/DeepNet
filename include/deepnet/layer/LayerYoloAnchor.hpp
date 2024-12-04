/// Copyright (c)2022 HanulSoft(HNS)

#pragma once

#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

class LayerYoloAnchor{

    int _end1;
    int _end2;
    int _str;
    float _grid_cell_offset;

  public:
    /// 생성자.
    LayerYoloAnchor(int end1, int end2, float grid_cell_offset, int stride);

    void forward();

    const TensorGpu& anchor() const { return _anchor; }
    const TensorGpu& stride() const { return _stride; }

  protected:
    TensorGpu _stride;
    TensorGpu _anchor;
};

} // namespace layer
} // namespace deepnet
