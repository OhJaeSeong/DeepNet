/// Copyright (c)2021 Electronics and Telecommunications Research
/// Institute(ETRI)

#include "deepnet/Debug.hpp"
#include "deepnet/Network.hpp"
#include "deepnet/Tensor.hpp"
#include "deepnet/layer/LayerConvolutional.hpp"
#include "deepnet/layer/LayerFullyConnected.hpp"
#include "deepnet/range.hpp"
#include "deepnet_test/DeepNetTest.hpp"

using namespace deepnet;

// TODO
DEEPNET_TEST_BEGIN(TestConvAndFC, !deepnet_test::autorun) {
    DEEPNET_TRACER;

    // 입력 값을 설정한다.
    TensorGpu x(2, 1, 7, 7);
    TensorCpu x_cpu(2, 1, 7, 7);
    x_cpu.set(range<float>(2 * 1 * 7 * 7).data());
    x.from(x_cpu);

    Network model(x);

    // conv 레이어를 생성한다.
    model += new layer::LayerConvolutional(x,                 //
                                           model.workspace(), //
                                           4, 3, 3);
    model += new layer::LayerConvolutional(model.y(),         //
                                           model.workspace(), //
                                           9, 3, 3);
    model += new layer::LayerFullyConnected(model.y(), 10);
    model += new layer::LayerFullyConnected(model.y(), 1);

    DEEPNET_ASSERT(model.size() == 4);

    model.print();
    model.randomizeWeight();

    // 전방향 전파를 실행한다.
    model.forward(x);
    model.debug();

    // 결과를 출력한다.
    TensorCpu y_cpu(model.y());
    y_cpu.print("y = ");
}
DEEPNET_TEST_END(TestConvAndFC)
