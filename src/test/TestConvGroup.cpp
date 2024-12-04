/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/LossMeanSquaredError.hpp"
#include "deepnet/Network.hpp"
#include "deepnet/Trainer.hpp"
#include "deepnet/Timer.hpp"
#include "deepnet/Weight.hpp"
#include "deepnet/layer/LayerConvolutional.hpp"
#include "deepnet/range.hpp"
#include "deepnet_test/DeepNetTest.hpp"

using namespace deepnet;

DEEPNET_TEST_BEGIN(TestLayerConvolutionalGroups, !deepnet_test::autorun) {
    TensorCpu x_cpu(1, 3, 4, 4);

    // 입력 값을 설정한다.
    x_cpu.set(range<float>(-12, -12 + 48));
    x_cpu.print("x = ");

    TensorGpu x(x_cpu);
    Network model(x);

    // conv 레이어를 생성한다.
    auto *l = new layer::LayerConvolutional( //
        x, model.workspace(), 2, 3, 3, 1, 0, true, 1);
    model += l;

    // 필터 값을 설정한다.
    TensorCpu w_cpu(l->w.dimension());
    TensorCpu b_cpu(l->b.dimension());

    for(int b = 0; b < 2; b += 1){
        for(int c = 0; c < 2; c += 1){
            for(int h = 0; h < 3; h += 1){
                if(b == 1){
                    w_cpu.at(b, c, h, 0) = 0.1;
                    w_cpu.at(b, c, h, 1) = 0.2;
                    w_cpu.at(b, c, h, 2) = 0.3;
                }else{
                    w_cpu.at(b, c, h, 0) = 0.1;
                    w_cpu.at(b, c, h, 1) = 0.3;
                    w_cpu.at(b, c, h, 2) = 0.1;
                }
            }
        }
    }
    // w_cpu.fill(0.1f);
    b_cpu.fill(0.0f);

    l->w.from(w_cpu);
    l->b.from(b_cpu);

    // model.print();

    // 전방향 전파를 실행한다.
    model.forward(x);

    TensorCpu y_cpu(model.y());
    y_cpu.print("y = ");
}
DEEPNET_TEST_END(TestLayerConvolutionalGroups)

/*
DEEPNET_TEST_BEGIN(TestLayerConvolutional1Layer, !deepnet_test::autorun) {
    TensorCpu x_cpu(1, 1, 3, 3);

    // 입력 값을 설정한다.
    x_cpu.set(range<float>(9));
    x_cpu.print("x = ");

    TensorGpu x(x_cpu);
    Network model(x);

    // conv 레이어를 생성한다.
    auto *l = new layer::LayerConvolutional(x, model.workspace(), //
                                            1, 3, 3, 1, 0);
    model += l;

    // 필터 값을 설정한다.
    TensorCpu w_cpu(l->w.dimension());
    TensorCpu b_cpu(l->b.dimension());

    w_cpu.set(range<float>(9));
    b_cpu.fill(0.0f);

    l->w.from(w_cpu);
    l->b.from(b_cpu);

    model.print();

    // 전방향 전파를 실행한다.
    model.forward(x);

    TensorCpu y_cpu(model.y());
    y_cpu.print("y = ");

    DEEPNET_ASSERT_EQ(y_cpu[0], 204.0f, 0.0001f);
}
DEEPNET_TEST_END(TestLayerConvolutional1Layer)


DEEPNET_TEST_BEGIN(TestLayerConvolutional3Layers, !deepnet_test::autorun) {
    // 입력 값을 설정한다.
    TensorCpu x_cpu(1, 1, 7, 7);
    x_cpu.set(range<float>(49));
    x_cpu.print("x = ");

    TensorGpu x(x_cpu);
    Network model(x);

    // conv 레이어를 생성한다.
    auto a = new layer::LayerConvolutional(x, model.workspace(), //
                                           1, 3, 3, 1, 0);
    auto b = new layer::LayerConvolutional(a->y(), model.workspace(), //
                                           1, 3, 3, 1, 0);
    auto c = new layer::LayerConvolutional(b->y(), model.workspace(), //
                                           1, 3, 3, 1, 0);

    // 필터 값을 설정한다.
    TensorCpu w_cpu(a->w.dimension());
    TensorCpu b_cpu(a->b.dimension());

    w_cpu.set(range<float>(9));
    b_cpu.fill(0.0f);

    a->w.from(w_cpu);
    a->b.from(b_cpu);
    b->w.from(w_cpu);
    b->b.from(b_cpu);
    c->w.from(w_cpu);
    c->b.from(b_cpu);

    model += a;
    model += b;
    model += c;

    DEEPNET_ASSERT(model.size() == 3);

    model.print();

    // 전방향 전파를 실행한다.
    model.forward(x);
    TensorCpu y_cpu(model.y());
    y_cpu.print("y = ");

    DEEPNET_ASSERT_EQ(y_cpu[0], 1632960.0f, 0.0001f);
}
DEEPNET_TEST_END(TestLayerConvolutional3Layers)
*/