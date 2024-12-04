/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/BaseCudnn.hpp"
#include "deepnet/Tensor.hpp"
#include <tuple>
#include <vector>

namespace deepnet {

namespace layer {
class Layer;
} // namespace layer

/// 텐서의 GPU 버전.
class TensorGpu : public Tensor {
    // CUDNN 텐서 기술자.
    cudnnTensorDescriptor_t _descriptor;

  private:
    /// 생성자.
    TensorGpu(const TensorGpu &tensor) {
        throw std::runtime_error("Copy operation prohibited!");
    }

  protected:
    /// 새 크기와 기존 크기와 다르면 메모리를 다시 할당한다.
    virtual void resize(size_t new_size) override;

  public:
    /// 생성자.
    TensorGpu() : Tensor(), _descriptor(nullptr) {
        SAFE_CUDNN(cudnnCreateTensorDescriptor(&_descriptor));
    }

    /// 생성자.
    TensorGpu(int batch, int channel, int height, int width) : TensorGpu() {
        setDimension(batch, channel, height, width);
    }

    /// 생성자.
    TensorGpu(Dimension dimension) : TensorGpu() {              //
        setDimension(dimension.batch(), dimension.channel(), //
                     dimension.height(), dimension.width());
    }

    /// 생성자.
    explicit TensorGpu(const TensorCpu &tensor);

    /// 소멸자.
    virtual ~TensorGpu() {
        if (_descriptor) {
            UNSAFE_CUDNN(cudnnDestroyTensorDescriptor(_descriptor));
            _descriptor = nullptr;
        }

        resize(0);
    }

    /// 값을 얻는다.
    inline const cudnnTensorDescriptor_t descriptor(void) const {
        return _descriptor;
    }

    /// 차원의 크기를 설정한다.
    virtual void setDimension(int batch, int height, int width,
                              int channel) override;

    /// 차원의 크기를 설정한다.
    virtual void setDimension(const Dimension &dimension) override {
        setDimension(dimension.batch(), dimension.channel(), //
                     dimension.height(), dimension.width());
    }

    /// 데이터 타입을 변경한다.
    void convertDataType(cudnnDataType_t data_type);

    /// 형 변환 복사 연산자.
    void from(const TensorCpu &t);

    /// 복사 연산자.
    void operator=(const TensorGpu &t);

    /// 전방향 레이어 연산을 실행한다.
    /// @return l->y()
    const TensorGpu &operator>>(layer::Layer *l) const;

    /// 역방향 레이어 연산을 실행한다.
    /// @return l->dx()
    const TensorGpu &operator<<(layer::Layer *l) const;
};

} // namespace deepnet
