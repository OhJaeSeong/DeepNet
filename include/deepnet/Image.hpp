/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Tensor.hpp"
#if FEATURE_USE_OPENCV == 1
#include <opencv2/opencv.hpp>

namespace deepnet {

/// 영상 클래스.
class Image {
  public:
    /// 생성자.
    Image() {}

    /// 영상 파일을 읽고 TensorCpu로 변환한다.
    /// 영상의 크기는 Tensor에 맞춰진다.
    static bool read(const char *file_path,               //
                     cv::Mat &original, cv::Mat &resized, //
                     TensorCpu &tensor, int batch = 0);

    /// 영상 파일을 읽고 TensorCpu로 변환한다.
    /// 영상의 크기는 Tensor에 맞춰진다.
    static bool read(const char *file_path, //
                     TensorCpu &tensor, int batch = 0, bool normalize = true);
};

} // namespace deepnet
#endif
