/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/TrainerTriplet.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/Image.hpp"
#include <cmath>
#if FEATURE_USE_OPENCV == 1
#include <opencv2/opencv.hpp>
#endif
#include <stdio.h>

namespace deepnet {

void TrainerTriplet::train(     //
    Network &model,             //
    DatasetTriplet &dataset,    //
    LossTriplet &loss_function, //
    float learning_rate) {
    DEEPNET_TRACER;

    std::cout << "Start Epoch = " << start_epoch << std::endl;
    std::cout << "End Epoch = " << end_epoch << std::endl;

    // 모델을 학습 모드로 설정한다.
    model.train();

    // 입출력 텐서를 준비한다.
    auto input_dimension = model.dx().dimension();
    TensorCpu x_cpu(input_dimension);
    TensorGpu x(input_dimension);

    auto output_dimension = model.y().dimension();
    TensorCpu y_cpu(output_dimension);
    TensorCpu dy_cpu(output_dimension);
    TensorGpu dy(output_dimension);

    // 세개의 배치가 anchor, positive, negative로 이루어져 있음.
    auto batch_size = x.batch() / 3;
    auto total_batch = (int)dataset.size() / batch_size;

    DEEPNET_LOG_TIME("Start training...");

    for (auto epoch = start_epoch; epoch <= end_epoch; epoch++) {
        float epoch_loss = 0.0f;
        float batch_loss = 0.0f;
        auto batch_count = 0;
        auto score = 0;

        dataset.shuffle();

        while (true) {
            // 학습 데이터를 읽는다.
            if (!dataset.get(x_cpu))
                break;

            x.from(x_cpu);
            
            batch_count++;

            // 전방향 전파를 실행한다.
            model.forward(x);
            // model.debug();
            y_cpu.from(model.y());

            // 손실값을 계산한다.
            auto loss = loss_function(y_cpu, dy_cpu);
            if (!isnan(loss)) {
                batch_loss += loss;
                epoch_loss += loss;
            }

            // 역방향 전파를 실행한다.
            dy.from(dy_cpu);
            model.backward(dy);

            // 모델의 가중치를 갱신한다.
            model.update(learning_rate);

            // 로그를 출력한다.
            if (log_duration_per_batch > 0 &&
                batch_count % log_duration_per_batch == 0) {
                std::cout << "Batch = " << batch_count << "/" << total_batch
                          << "(" << (batch_count * 100 / total_batch) << "%)"
                          << ", loss = "
                          << (batch_loss / log_duration_per_batch) << "/"
                          << (epoch_loss / batch_count) << std::endl;

                batch_loss = 0.0f;

                if (Trainer::isFileExist("stop.txt"))
                    break;
            }
        }

        if (log_duration_per_epoch > 0 && epoch % log_duration_per_epoch == 0) {
            std::cout << "Epoch = " << epoch << "/" << end_epoch   //
                      << "(" << (epoch * 100 / end_epoch) << "%)"  //
                      << ", loss = " << (epoch_loss / batch_count) //
                      << std::endl;
        }

        if (save_duration_per_epoch > 0 &&
            epoch % save_duration_per_epoch == 0) {
            std::string weight_file_name = output_weight_path      //
                                           + "model-"              //
                                           + std::to_string(epoch) //
                                           + ".weight";

            DEEPNET_LOG_TIME("Start writing " << weight_file_name);

            model.writeWeight(weight_file_name.c_str(), format);
        }

        if (target_loss > 0.0f && target_loss > (epoch_loss / batch_count)) {
            std::cout << "Epoch = " << epoch << "/" << end_epoch   //
                      << "(" << (epoch * 100 / end_epoch) << "%)"  //
                      << ", loss = " << (epoch_loss / batch_count) //
                      << " < " << target_loss << std::endl;

            break;
        }

        if (Trainer::isFileExist("stop.txt"))
            break;
    }

    DEEPNET_LOG_TIME("End training...");
}

} // namespace deepnet
