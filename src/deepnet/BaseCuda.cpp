/// Copyright (c)2021 Electronics and Telecommunications Research
/// Institute(ETRI)

#include "deepnet/BaseCuda.hpp"

namespace deepnet {

/// CUDNN 핸들.
cublasHandle_t BaseCuda::cublas_handle = nullptr;

const char *BaseCuda::cublasGetErrorString(cublasStatus_t status) {
    /// CUBLAS 오류 코드를 문자열로 반환한다.

    if (status == CUBLAS_STATUS_SUCCESS)
        return "CUBLAS_STATUS_SUCCESS";
    else if (status == CUBLAS_STATUS_NOT_INITIALIZED)
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    else if (status == CUBLAS_STATUS_ALLOC_FAILED)
        return "CUBLAS_STATUS_ALLOC_FAILED";
    else if (status == CUBLAS_STATUS_INVALID_VALUE)
        return "CUBLAS_STATUS_INVALID_VALUE";
    else if (status == CUBLAS_STATUS_ARCH_MISMATCH)
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    else if (status == CUBLAS_STATUS_MAPPING_ERROR)
        return "CUBLAS_STATUS_MAPPING_ERROR";
    else if (status == CUBLAS_STATUS_EXECUTION_FAILED)
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    else if (status == CUBLAS_STATUS_INTERNAL_ERROR)
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    else if (status == CUBLAS_STATUS_NOT_SUPPORTED)
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    else if (status == CUBLAS_STATUS_LICENSE_ERROR)
        return "CUBLAS_STATUS_LICENSE_ERROR";
    else
        return "Unknown error";
}

} // namespace deepnet
