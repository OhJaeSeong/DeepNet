/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerYoloAnchor.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>
#include <iostream>

#define USE_CUDNN

// For CUDNN API,
// See https://docs.nvidia.com/deeplearning/cudnn/api/index.html

void gpu_yoloanchor_forward(float *anchor, float *stride, int end1, int end2, float grid_cell_offset, int str);

namespace deepnet {
namespace layer {

/// 생성자.
LayerYoloAnchor::LayerYoloAnchor(int end1, int end2, float grid_cell_offset, int str) 
    // number 분할갯수, dum 분할하는 차원, count y값으로 반환할 값(몇번째 것을 반환?)
    : _end1(end1), _end2(end2), _grid_cell_offset(grid_cell_offset), _str(str){
    DEEPNET_TRACER;
     _anchor.setDimension(1, end1 * end2, 2, 1);
     _stride.setDimension(1, end1 * end2, 4, 1);

    // DEEPNET_ASSERT(_anchor.isEmpty());
}

void LayerYoloAnchor::forward() {
    DEEPNET_TRACER;
    try {
        gpu_yoloanchor_forward(_anchor.data(), _stride.data(), _end1, _end2, _grid_cell_offset, _str);

    } catch (std::exception &e) {
        DEEPNET_LOG("Error = " << e.what());

        throw e;
    }
}

} // namespace layer
} // namespace deepnet
