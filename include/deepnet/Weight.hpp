/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Tensor.hpp"

namespace deepnet {
/// 가중치 관련 클래스.
class Weight {
  public:
    /// 가중치 초기화 방법.
    enum class InitMethod {
        Xavier, Xavier2, Fill01, FillZero
    };

    /// 가중치 갱신 방법.
    enum class UpdateMethod {
        SGD,
    };

    /// 가중치 파일 저장 형식.
    enum class Format { Torch, Darknet };

    /// 네트워크 정보를 출력한다.
    /// @param filter_gpu 필터(out_channel, height, width, ini_channel).
    /// @param method 가중치 초기화 방법(기본값 = InitMethod::Xavier).
    static void initializeWeight(TensorGpu &filter_gpu,
                                 InitMethod method = InitMethod::Xavier);

    /// 가중치 정보를 읽는다.
    static void readWeight(FILE *file, TensorGpu &tensor, Format format);

    /// 가중치 정보를 저장한다.
    static void writeWeight(FILE *file, const TensorGpu &tensor, Format format);
};
} // namespace deepnet
