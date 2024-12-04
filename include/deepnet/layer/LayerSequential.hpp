/// Copyright (c)2021 Electronics and Telecommunications Research
/// Institute(ETRI)

#pragma once

#include "deepnet/layer/LayerWeighted.hpp"
#include <vector>

namespace deepnet {
namespace layer {

/// 여러 개의 레이어를 포함하는 레이어.

/// LayerSequential은 자체적인 가중치 값은 가지고 있지 않지만,
/// 서브 레이어에서 가중치 값을 가질 수 있으므로 LayerWeighted에서
/// 상속을 받는다.
class LayerSequential : public LayerWeighted {
  protected:
    /// 레이어.
    std::vector<Layer *> _layers;

  public:
    LayerSequential(const TensorGpu &x) : LayerWeighted(x) {}

    /// 소멸자.
    virtual ~LayerSequential() {
        for (auto *l : _layers)
            delete l;

        _layers.clear();
    }

    /// 레이어 타입 이름.
    virtual const char *type(void) const override { return "Sequential"; }

    /// 학습 상태 여부를 설정한다.
    virtual void train(void) override;

    /// 학습 상태 여부를 설정한다.
    virtual void eval(void) override;

    /// 레이어의 수.
    inline int size(void) const { return (int)_layers.size(); }

    /// 지정한 인덱스의 레이어를 반환한다.
    inline Layer *operator[](int index) {
        if (index < 0 || _layers.size() <= index)
            throw std::runtime_error("Out of index");

        return _layers[index];
    }

    /// 레이어를 추가한다.
    virtual void operator+=(Layer *layer) { _layers.push_back(layer); }

    /// 전방향 전파.
    /// 전파 결과는 마지막 층의 _y에 저장된다.
    virtual void forward(const TensorGpu &x) override;

    /// 역방향 전파.
    /// dy = y - target.
    /// 전파 결과는 첫번째 층의 _dx에 저장된다.
    virtual void backward(const TensorGpu &dy) override;

    /// 가중치 정보를 초기화한다.
    virtual void
    randomizeWeight(Weight::InitMethod method = Weight::InitMethod::Xavier) override;

    /// 가중치 정보를 읽는다.
    virtual void readWeight(FILE *file, Weight::Format format) override;

    /// 가중치 정보를 저장한다.
    virtual void writeWeight(FILE *file, Weight::Format format) const override;

    /// 출력값을 반환한다.
    virtual const TensorGpu &y(void) const override;

    /// 델타 x 값을 반환한다.
    virtual const TensorGpu &dx(void) const override;

    /// 레이어 정보를 콘솔에 출력한다.
    virtual void print(tool::TablePrinter &printer, int depth = 0,
                       int index = 1) const override;

    /// 네트워크의 가중치 정보를 출력한다.
    virtual void printWeight(tool::TablePrinter &printer, int depth = 0,
                             int index = 1) const override;

    /// 레이어 출력 정보를 콘솔에 출력한다.
    virtual void debug(tool::TablePrinter &printer, int depth = 0,
                       int index = 1) override;

    /// 가중치 정보를 갱신한다.
    virtual void
    update(float learning_rate,
           Weight::UpdateMethod method = Weight::UpdateMethod::SGD) override;

    /// 가중치 값의 체크섬을 계산한다.
    virtual unsigned char checksum(void) const override;
};

} // namespace layer
} // namespace deepnet
