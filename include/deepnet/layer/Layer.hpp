/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/BaseCudnn.hpp"
#include "deepnet/Tensor.hpp"
#include "deepnet/tool/TablePrinter.hpp"
#include <string>

namespace deepnet {
namespace layer {

/// 가중치를 포함하지 않는 레이어의 추상 클래스.
class Layer : public BaseCudnn {
    /// 복사 생성자를 금지한다.
    Layer() : _px(nullptr), _training(false) { DEEPNET_ASSERT(cudnn_handle); }

    /// 복사 연산을 금지한다.
    void operator=(const Layer &) {
        DEEPNET_ASSERT(!"Copy operation prohibited!");
    }

  protected:
    /// 입력값의 포인터.
    const TensorGpu *_px;

    /// 출력값.
    TensorGpu _y;

    /// 입력값의 차분값.
    TensorGpu _dx;

    /// 학습 상태 여부.
    bool _training;

  public:
    /// 생성자.
    Layer(const TensorGpu &x);

    /// 소멸자.
    virtual ~Layer(){};

    /// 레이어 타입 이름.
    virtual const char *type(void) const = 0;

    /// 학습 상태 여부를 설정한다.
    virtual void train(void);

    /// 학습 상태 여부를 설정한다.
    virtual void eval(void);

    /// 전방향 전파.
    /// 전방향 전파 결과는 _y에 저장된다.
    virtual void forward(const TensorGpu &x);

    /// 역방향 전파.
    /// dy = y - target.
    /// 역방향 전파 결과는 _dx에 저장된다.
    virtual void backward(const TensorGpu &dy);

    /// 출력값을 반환한다.
    virtual const TensorGpu &y(void) const;

    /// 델타 x 값을 반환한다.
    virtual const TensorGpu &dx(void) const;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter &printer, //
                       int depth = 0, int index = 1) const;

    /// 네트워크의 가중치 정보를 출력한다.
    virtual void printWeight(tool::TablePrinter &printer, //
                       int depth = 0, int index = 1) const;

    /// 레이어 출력 정보를 콘솔에 출력한다.
    virtual void debug(tool::TablePrinter &printer, //
                       int depth = 0, int index = 1);

    /// 레이어 출력 정보를 콘솔에 출력한다.
    void debugOutput(tool::TablePrinter &printer, TensorCpu &output, //
                     int depth = 0, int index = 1);
};

} // namespace layer
} // namespace deepnet
