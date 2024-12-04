/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerYolo.hpp"
#include "deepnet/Debug.hpp"

void gpu_yolo_forward(float *y, float *x,                            //
                      int batch, int channel, int height, int width, //
                      int image_height, int image_width,             //
                      int candidates, int classes,                   //
                      float *anchors);

namespace deepnet {
namespace layer {

LayerYolo::LayerYolo(const TensorGpu &x,                //
                     int image_height, int image_width, //
                     int candidates, int classes,       //
                     std::vector<std::pair<int, int>> &anchors)
    : Layer(x), _image_height(image_height), _image_width(image_width), //
      _candidates(candidates), _classes(classes) {
    DEEPNET_TRACER;

    // TODO: NCHW
    DEEPNET_ASSERT(0);

    DEEPNET_ASSERT(image_height > 0);
    DEEPNET_ASSERT(image_width > 0);
    DEEPNET_ASSERT(candidates > 0);
    DEEPNET_ASSERT(classes > 0);
    DEEPNET_ASSERT(anchors.size() == candidates);
    DEEPNET_ASSERT(x.channel() == candidates * (5 + classes));

    // anchor의 값을 GPU에 복사한다.
    auto buffer = new float[2 * candidates];

    for (auto i = 0; i < candidates; i++) {
        buffer[i * 2] = (float)anchors[i].first;
        buffer[i * 2 + 1] = (float)anchors[i].second;
    }

    SAFE_CUDA(cudaMalloc((void **)&_anchors, //
                         2 * candidates * sizeof(float)));

    // 디바이스 메모리를 복사한다.
    SAFE_CUDA(cudaMemcpy((void *)_anchors,               //
                         (void *)buffer,                 //
                         2 * candidates * sizeof(float), //
                         cudaMemcpyHostToDevice));
    delete[] buffer;

    _y.setDimension(x.dimension());
}

LayerYolo::~LayerYolo() {
    if (_anchors) {
        UNSAFE_CUDA(cudaFree((void *)_anchors));
        _anchors = nullptr;
    }
}

void LayerYolo::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);

    gpu_yolo_forward(_y.data(), x.data(),                                //
                     x.batch(), x.channel(), x.height(), x.width(),      //
                     _image_height, _image_width, _candidates, _classes, //
                     _anchors);
};

void LayerYolo::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    Layer::backward(dy);

    DEEPNET_ASSERT(!"Not implemented");

    // gpu_yolo_backward(_dx.data(), dy.data(), dy.size(), //
    //                   _height, _width, _candidates, _classes, _anchors);
};

} // namespace layer
} // namespace deepnet
