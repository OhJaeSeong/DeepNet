/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Tensor.hpp"
#include <tuple>
#include <vector>

#if FEATURE_USE_OPENCV == 1
namespace cv {
class Mat;
} // namespace cv
#endif

namespace deepnet {

class TensorGpu;

/// 텐서의 CPU 버전.
class TensorCpu : public Tensor {
    /// 생성자.
    TensorCpu(const TensorCpu &tensor) {
        throw std::runtime_error("Copy operation prohibited!");
    }

  protected:
    /// 새 크기와 기존 크기와 다르면 메모리를 다시 할당한다.
    virtual void resize(size_t new_size) override;

  public:
    /// 생성자.
    TensorCpu() : Tensor() {}

    /// 생성자.
    TensorCpu(int batch, int channel, int height, int width) : TensorCpu() {
        setDimension(batch, channel, height, width);
    }

    /// 생성자.
    TensorCpu(Dimension dimension) : TensorCpu() {
        setDimension(dimension.batch(), dimension.channel(), //
                     dimension.height(), dimension.width());
    }

    /// 생성자.
    explicit TensorCpu(const TensorGpu &tensor);

    /// 소멸자.
    virtual ~TensorCpu() { resize(0); }

    /// 차원의 크기를 설정한다.
    virtual void setDimension(int batch, int channel, int height,
                              int width) override;

    /// 차원의 크기를 설정한다.
    virtual void setDimension(const Dimension &dimension) override {
        setDimension(dimension.batch(), dimension.channel(), //
                     dimension.height(), dimension.width());
    }

    /// 형 변환 복사 연산자.
    void from(const TensorGpu &t);

#if FEATURE_USE_OPENCV == 1
    /// CV 영상을 텐서로 변환한다.
    void from(const cv::Mat &mat, int batch_index = 0);

    /// 텐서를 CV 영상으로 변환한다.
    void to(cv::Mat &mat, int batch_index = 0) const;
#endif
    /// 값 채우기.
    void fill(float value);

    /// 값을 초기화한다.
    void set(std::vector<float> values);

    /// 값을 초기화한다.
    void set(const float *const values);

    /// 값 얻기.
    inline float &operator[](size_t index) { return _data[index]; }

    /// 값 얻기.
    inline float &at(int b, int c, int h, int w) {
        return _data[((b * _channel + c) * _height + h) * _width + w];
    }

    /// 값 얻기.
    inline float at(int b, int c, int h, int w) const {
        return _data[((b * _channel + c) * _height + h) * _width + w];
    }

    /// 값 곱하기.
    void operator *=(float value);

    /// 값을 출력한다.
    void print(const char *prefix = "", const char *postfix = "\n", //
               const char *bsep = ", ", const char *csep = ", ",    //
               const char *hsep = ", ", const char *wsep = ", ") const;

#if FEATURE_USE_OPENCV == 1
    /// 텐서 값을 영상으로 표시한다.
    void show(const char *title) const;

    /// 텐서 값을 영상으로 표시한다.
    void show(const char *title, int batch_index) const;
#endif
    /// 텐서의 최소값과 최대값을 반환한다.
    std::tuple<float, float> getMinMax(void) const;

    /// 텐서 값의 체크섬을 계산한다.
    unsigned char checksum(void) const;
};

} // namespace deepnet
