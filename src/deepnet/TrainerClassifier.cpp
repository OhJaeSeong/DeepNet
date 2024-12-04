/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/TrainerClassifier.hpp"
#include "deepnet/Debug.hpp"

namespace deepnet {

void TrainerClassifier::train(  //
    Network &model,             //
    DatasetSupervised &dataset, //
    Loss &loss_function,       //
    float learning_rate) {
    DEEPNET_TRACER;

    std::cout << "Start Epoch = " << start_epoch << std::endl;
    std::cout << "End Epoch = " << end_epoch << std::endl;

    // 모델을 학습 모드로 설정한다.
    model.train();

    // 입출력 텐서를 준비한다.
    auto input_dimension = model.dx().dimension();
    TensorCpu x_cpu(input_dimension);
    TensorGpu x_gpu(input_dimension);
    auto output_dimension = model.y().dimension();
    TensorCpu target_cpu(output_dimension);
    TensorCpu y_cpu(output_dimension);
    TensorGpu dy(output_dimension);
    TensorCpu dy_cpu(output_dimension);

    auto batch_size = input_dimension.batch();
    auto total_batch = (int)dataset.size() / batch_size;

    DEEPNET_LOG_TIME("Start training...");

    for (auto epoch = start_epoch; epoch <= end_epoch; epoch++) {
        float total_loss = 0.0f;
        float loss = 0.0f;
        auto batch_count = 0;
        auto score = 0;

        dataset.shuffle();

        while (true) {
            // 학습 데이터를 읽는다.
            if (!dataset.get(x_cpu, target_cpu))
                break;

            batch_count++;
            x_gpu.from(x_cpu);

            // 전방향 전파를 실행한다.
            model.forward(x_gpu);
            y_cpu.from(model.y());

            // 손실값을 계산한다.
            auto l = loss_function(y_cpu, target_cpu, dy_cpu);
            loss += l;
            total_loss += l;

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
                          << ", loss = " << (loss / log_duration_per_batch)
                          << "/" << (total_loss / batch_count) << std::endl;

                loss = 0.0f;

                if (Trainer::isFileExist("stop.txt"))
                    break;
            }
        }

        if (log_duration_per_epoch > 0 && epoch % log_duration_per_epoch == 0) {
            std::cout << "Epoch = " << epoch << "/" << end_epoch   //
                      << "(" << (epoch * 100 / end_epoch) << "%)"  //
                      << ", loss = " << (total_loss / batch_count) //
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

        if (target_loss > 0.0f && target_loss > (total_loss / batch_count)) {
            std::cout << "Epoch = " << epoch << "/" << end_epoch   //
                      << "(" << (epoch * 100 / end_epoch) << "%)"  //
                      << ", loss = " << (total_loss / batch_count) //
                      << " < " << target_loss << std::endl;

            break;
        }

        if (Trainer::isFileExist("stop.txt"))
            break;
    }

    DEEPNET_LOG_TIME("End training...");
}

} // namespace deepnet
