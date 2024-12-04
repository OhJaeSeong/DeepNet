/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/BaseCuda.hpp"
#include "deepnet/Debug.hpp"
#include <cudnn.h>
#include <stdexcept>

#ifdef _DEBUG

#define SAFE_CUDNN(statement)                                                  \
    {                                                                          \
        auto status = statement;                                               \
        if (status) {                                                          \
            deepnet::Tracer::printStack();                                     \
            DEEPNET_LOG(#statement);                                              \
            throw std::runtime_error(cudnnGetErrorString(status));             \
        }                                                                      \
    }

#define UNSAFE_CUDNN(statement)                                                \
    {                                                                          \
        auto status = statement;                                               \
        if (status) {                                                          \
            deepnet::Tracer::printStack();                                     \
            DEEPNET_LOG(#statement);                                              \
        }                                                                      \
    }

#else // _DEBUG

#define SAFE_CUDNN(statement)                                                  \
    {                                                                          \
        auto status = statement;                                               \
        if (status) {                                                          \
            DEEPNET_LOG(#statement);                                              \
            throw std::runtime_error(cudnnGetErrorString(status));             \
        }                                                                      \
    }

#define UNSAFE_CUDNN(statement)                                                \
    {                                                                          \
        auto status = statement;                                               \
        if (status) {                                                          \
            DEEPNET_LOG(#statement);                                              \
        }                                                                      \
    }

#endif // _DEBUG

namespace deepnet {

/// CUDNN 관련 기본 기능을 제공하는 클래스.
class BaseCudnn : public BaseCuda {
  public:
    /// CUDNN 핸들.
    static cudnnHandle_t cudnn_handle;

    /// 핸들을 생성한다.
    inline static void create(void) { SAFE_CUDNN(cudnnCreate(&cudnn_handle)); }

    /// 핸들을 삭제한다.
    inline static void destroy(void) { SAFE_CUDNN(cudnnDestroy(cudnn_handle)); }
};

} // namespace deepnet
