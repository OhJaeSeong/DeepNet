/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Debug.hpp"
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <stdexcept>

#ifdef _DEBUG

#define SAFE_CUDA(statement)                                                   \
    {                                                                          \
        auto error = statement;                                                \
        if (error) {                                                           \
            deepnet::Tracer::printStack();                                     \
            DEEPNET_LOG(#statement);                                              \
            throw std::runtime_error(cudaGetErrorString(error));               \
        }                                                                      \
    }

#define UNSAFE_CUDA(statement)                                                 \
    {                                                                          \
        auto error = statement;                                                \
        if (error) {                                                           \
            deepnet::Tracer::printStack();                                     \
            DEEPNET_LOG(#statement);                                              \
        }                                                                      \
    }

#define SAFE_CUBLAS(statement)                                                 \
    {                                                                          \
        auto error = statement;                                                \
        if (error) {                                                           \
            deepnet::Tracer::printStack();                                     \
            DEEPNET_LOG(#statement);                                              \
            throw std::runtime_error(                                          \
                deepnet::BaseCuda::cublasGetErrorString(error));               \
        }                                                                      \
    }

#define UNSAFE_CUBLAS(statement)                                               \
    {                                                                          \
        auto error = statement;                                                \
        if (error) {                                                           \
            deepnet::Tracer::printStack();                                     \
            DEEPNET_LOG(#statement);                                              \
        }                                                                      \
    }

#else // _DEBUG

#define SAFE_CUDA(statement)                                                   \
    {                                                                          \
        auto error = statement;                                                \
        if (error) {                                                           \
            DEEPNET_LOG(#statement);                                              \
            throw std::runtime_error(cudaGetErrorString(error));               \
        }                                                                      \
    }

#define UNSAFE_CUDA(statement)                                                 \
    {                                                                          \
        auto error = statement;                                                \
        if (error) {                                                           \
            DEEPNET_LOG(#statement);                                              \
        }                                                                      \
    }

#define SAFE_CUBLAS(statement)                                                 \
    {                                                                          \
        auto error = statement;                                                \
        if (error) {                                                           \
            DEEPNET_LOG(#statement);                                              \
            throw std::runtime_error(                                          \
                deepnet::BaseCuda::cublasGetErrorString(error));               \
        }                                                                      \
    }

#define UNSAFE_CUBLAS(statement)                                               \
    {                                                                          \
        auto error = statement;                                                \
        if (error) {                                                           \
            DEEPNET_LOG(#statement);                                              \
        }                                                                      \
    }

#endif // _DEBUG

namespace deepnet {

/// CUDA 관련 기본 함수를 제공한다.
class BaseCuda {
  public:
    /// CUBLAS 핸들.
    static cublasHandle_t cublas_handle;

    /// CUBLAS 상태 값의 오류 문자열을 반환한다.
    static const char *cublasGetErrorString(cublasStatus_t status);

    /// 핸들을 생성한다.
    inline static void create(void) {
        SAFE_CUBLAS(cublasCreate(&cublas_handle));
    }

    /// 핸들을 삭제한다.
    inline static void destroy(void) {
        SAFE_CUBLAS(cublasDestroy(cublas_handle));
    }
};
} // namespace deepnet
