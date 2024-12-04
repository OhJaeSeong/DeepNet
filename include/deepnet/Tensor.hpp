/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include <stdexcept>
#include <tuple>
#include <vector>
#include "deepnet/Features.hpp"

namespace deepnet {

/// 텐서의 차원 크기.
class Dimension {
  protected:
    /// 텐서의 배치 크기.
    int _batch;
    /// 텐서의 채널 수.
    int _channel;
    /// 텐서의 높이.
    int _height;
    /// 텐서의 폭.
    int _width;

  public:
    /// 생성자.
    Dimension() : _batch(0), _channel(0), _height(0), _width(0) {}

    /// 생성자.
    Dimension(int batch, int channel, int height, int width)
        : _batch(batch), _channel(channel), //
          _height(height), _width(width) {}

    /// 생성자.
    Dimension(const Dimension &dim)
        : _batch(dim._batch), _channel(dim._channel), //
          _height(dim._height), _width(dim._width) {}

    /// 배치 크기.
    inline int batch(void) const { return _batch; }

    /// 채널 크기.
    inline int channel(void) const { return _channel; }

    /// 높이.
    inline int height(void) const { return _height; }

    /// 폭.
    inline int width(void) const { return _width; }

    /// 비교 연산자.
    inline bool operator==(const Dimension &dim) const {
        return _batch == dim._batch &&     //
               _channel == dim._channel && //
               _height == dim._height &&   //
               _width == dim._width;
    }

    /// 텐서의 차원 값을 문자열로 변환한다.
    operator std::string() const;
};

/// 텐서.
class Tensor : public Dimension {
    /// 복사 연산을 금지한다.
    void operator=(const Tensor &) {
        throw std::runtime_error("Copy operation prohibited!");
    }

  protected:
    /// 배열의 수(= batch * channel * height * width).
    size_t _size;

    /// 메모리에 저장하는 데이터의 포인터.
    float *_data;

    /// 새 크기와 기존 크기와 다르면 메모리를 다시 할당한다.
    virtual void resize(size_t new_size) = 0;

  public:
    /// 생성자.
    Tensor(void);

    /// 소멸자.
    virtual ~Tensor(){};

    /// 텐서가 비어 있으면 true를 반환한다.
    inline bool isEmpty(void) const {
        return _batch < 1 || _channel < 1 || _height < 1 || _width < 1 ||
               _size < 1 || _data == 0;
    }

    /// 차원의 크기를 얻는다.
    inline const Dimension& dimension(void) const {
        return *this;
    }

    /// 데이터 수(= batch * channel * height width).
    inline size_t size(void) const { return _size; }

    /// 값을 얻는다.
    inline float *data(void) const { return _data; }

    /// 차원의 크기를 설정한다.
    virtual void setDimension(int batch, int channel, int height,
                              int width) = 0;

    /// 차원의 크기를 설정한다.
    virtual void setDimension(const Dimension &dimension) = 0;

    /// 차원의 크기를 조정한다.
    void reshape(int batch, int channel, int height, int width);

    /// 차원의 크기를 조정한다.
    void reshape(const Dimension &dimension);
}; // namespace deepnet

} // namespace deepnet

#include "deepnet/TensorCpu.hpp"
#include "deepnet/TensorGpu.hpp"
