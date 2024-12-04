/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/LossTriplet.hpp"
#include "deepnet/Network.hpp"
#include "deepnet/Trainer.hpp"
#include "deepnet/dataset/DatasetTriplet.hpp"

namespace deepnet {

/// 학습기 클래스.
class TrainerTriplet : public Trainer {
  public:
    /// 생성자.
    inline TrainerTriplet() {}

    /// 모델을 학습한다.
    ///
    /// @param model 네트워크 모델.
    /// @param dataset 학습할 데이터셋.
    /// @param loss_function 손실 함수.
    /// @param learning_rate 학습률.
    void train(                     //
        Network &model,             //
        DatasetTriplet &dataset,    //
        LossTriplet &loss_function, //
        float learning_rate);
};

} // namespace deepnet
