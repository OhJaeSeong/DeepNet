/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Weight.hpp"
#include "deepnet/layer/Layer.hpp"

namespace deepnet {
namespace layer {

/// 가중치를 포함하는 레이어의 추상 클래스.
class LayerWeighted : public Layer {
    /// 바이어스 값 사용 여부.
    bool _bias;

  protected:
    /// 가중치 w에 대한 정보를 저항한다.
    cudnnFilterDescriptor_t _w_desc;
    /// 컨볼루션에 대한 정보를 저장한다.
    cudnnConvolutionDescriptor_t _conv_desc;

  protected:
  public:
    /// 가중치.
    TensorGpu w;
    /// 바이어스 항.
    TensorGpu b;

    /// 가중치의 델타값.
    TensorGpu dw;
    /// 바이어스 항의 델타값.
    TensorGpu db;

  public:
    /// 생성자.
    
    // /// LayerSequential에서 필요하다.
    // LayerWeighted() : _w_desc(nullptr), _conv_desc(nullptr), _bias(false) {}

    /// 생성자.
    LayerWeighted(const TensorGpu &x, bool bias = true)
        : Layer(x), _bias(bias), //
          _w_desc(nullptr), _conv_desc(nullptr) {}

    /// 소멸자.
    virtual ~LayerWeighted();

    /// 레이어 타입 이름.
    virtual const char *type(void) const override = 0;

    /// 가중치 정보를 읽는다.
    virtual void readWeight(FILE *file, Weight::Format format);

    /// 가중치 정보를 저장한다.
    virtual void writeWeight(FILE *file, Weight::Format format) const;

    /// 네트워크의 가중치 정보를 출력한다.
    virtual void printWeight(tool::TablePrinter &printer, //
                       int depth = 0, int index = 1) const;

    /// 가중치 정보를 초기화한다.
    virtual void randomizeWeight(Weight::InitMethod method);

    /// 바이어스 항 사용 여부를 반환한다.
    inline bool bias(void) const { return _bias; }

    /// 가중치 정보를 갱신한다.
    virtual void update(float learning_rate, Weight::UpdateMethod method=Weight::UpdateMethod::SGD);

    /// 가중치 값의 체크섬을 계산한다.
    virtual unsigned char checksum(void) const;
};

} // namespace layer
} // namespace deepnet
