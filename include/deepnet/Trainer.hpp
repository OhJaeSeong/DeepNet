/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Weight.hpp"

namespace deepnet {

/// 학습기 클래스.
class Trainer {
  public:
    /// 시작 주기. 기본값은 1.
    int start_epoch;

    /// 종료 주기. 기본값은 100.
    int end_epoch;

    /// 목표 손실값. 0이면 손실값 조건으로 종료 안함. 기본값은 0.
    float target_loss;

    /// 로그 메시지 출력 주기. 0이면 출력 안함. 기본값은 0.
    int log_duration_per_batch;

    /// 로그 메시지 출력 주기. 0이면 출력 안함. 기본값은 0.
    int log_duration_per_epoch;

    /// 가중치 파일 저장 주기. 0이면 저장 안함. 기본값은 0.
    int save_duration_per_epoch;

    /// 저장할 가중치 파일 경로.
    std::string output_weight_path;

    /// 로그 메시지 저장 파일 경로.
    std::string log_file_path;

    /// 가중치 파일 저장 형식.
    Weight::Format format;

  public:
    /// 생성자.
    inline Trainer()
        : start_epoch(1),            //
          end_epoch(100),            //
          target_loss(0.0f),         //
          log_duration_per_batch(0), //
          log_duration_per_epoch(0), //
          save_duration_per_epoch(0), format(Weight::Format::Darknet) {}

    /// 해당 경로의 파일이 있는지 확인한다.
    static bool isFileExist(const char *file_path);

    /// 값이 최대인 인덱스를 반환한다.
    static int argmax(const float *p, int size);

    /// 배치별로 텐서 값(CPU)이 최대인 인덱스를 반환한다.
    static std::vector<int> argmax(const TensorCpu &x);
};

} // namespace deepnet
