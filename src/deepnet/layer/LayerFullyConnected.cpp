/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerFullyConnected.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>

// For CUDNN API,
// See https://docs.nvidia.com/deeplearning/cudnn/api/index.html

namespace deepnet {
namespace layer {

/// 생성자.
LayerFullyConnected::LayerFullyConnected(const TensorGpu &x, int channel,
                                         bool bias)
    : LayerWeighted(x, bias), _out_channel(channel) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!x.isEmpty());

    int batch = x.batch();
    int in_channel = x.channel() * x.height() * x.width();

    // 가중치 w.
    w.setDimension(_out_channel, in_channel, 1, 1);

    // 출력값 y의 차원을 설정한다.
    _y.setDimension(batch, _out_channel, 1, 1);

    // 바이어스 항.
    if (bias) {
        b.setDimension(1, _out_channel, 1, 1);

        // 1-벡터. 역전파 연산에 사용한다.
        ones.setDimension(1, 1, 1, batch);
        TensorCpu temp(1, 1, 1, batch);
        temp.fill(1.0f);
        ones.from(temp);
    }
}

void LayerFullyConnected::train(void) {
    LayerWeighted::train();

    dw.setDimension(w.dimension());

    if (bias())
        db.setDimension(b.dimension());
}

void LayerFullyConnected::eval(void) {
    LayerWeighted::eval();

    dw.setDimension(0, 0, 0, 0);

    if (bias())
        db.setDimension(0, 0, 0, 0);
}

void LayerFullyConnected::mul(TensorGpu &c, const TensorGpu &a,
                              const TensorGpu &b, bool transa, bool transb) {
    DEEPNET_TRACER;

    // c(b,       h,w,c) = a(b,       h,w,c) * b(b,       h,w,c)
    // c(<batch>, <out>) = a(<batch>, <in> ) * b(<in>,    <out>)
    // c(m,       n    ) = a(m,       k    ) * b(k,       n    )
    auto c1 = c.batch();
    auto c2 = c.height() * c.width() * c.channel();
    auto a1 = a.batch();
    auto a2 = a.height() * a.width() * a.channel();
    auto b1 = b.batch();
    auto b2 = b.height() * b.width() * b.channel();

    DEEPNET_ASSERT(c1 == (transa ? a2 : a1));                 // <batch>
    DEEPNET_ASSERT(c2 == (transb ? b1 : b2));                 // <out>
    DEEPNET_ASSERT((transa ? a1 : a2) == (transb ? b2 : b1)); // <in>

    float alpha = 1.0f, beta = 0.0f;
    int m = c1;               // batch size.
    int n = c2;               // output size.
    int k = transa ? a1 : a2; // input size.

    // CUBLAS는 column-major이므로 a와 b의 위치를 바꾼다.
    SAFE_CUBLAS(cublasSgemm(                //
        cublas_handle,                      //
        transb ? CUBLAS_OP_T : CUBLAS_OP_N, //
        transa ? CUBLAS_OP_T : CUBLAS_OP_N, //
        n, m, k,                            //
        &alpha,                             //
        b.data(), b2,                       //
        a.data(), a2,                       //
        &beta,                              //
        c.data(), c2));
}

void LayerFullyConnected::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!w.isEmpty());

    Layer::forward(x);

    // y(b, o) = x(b, i) * w(o, i)^t.
    mul(_y, x, w, false, true);

    if (bias()) {
        float alpha = 1.0f;

        // y += b.
        SAFE_CUDNN(cudnnAddTensor( //
            cudnn_handle,          //
            &alpha,                //
            b.descriptor(),        //
            b.data(),              //
            &alpha,                //
            _y.descriptor(),       //
            _y.data()));
    }
}

void LayerFullyConnected::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!w.isEmpty());
    DEEPNET_ASSERT(!dw.isEmpty());

    Layer::backward(dy);

    // dw(o, i) = dy(b, o)^t * x(b, i).
    mul(dw, dy, *_px, true);

    if (bias()) {
        DEEPNET_ASSERT(!b.isEmpty());
        DEEPNET_ASSERT(!db.isEmpty());

        // db(1, o) = 1(1, b) * dy(b, o).
        mul(db, ones, dy);
    }

    // dx(b, i) = dy(b, o) * w(o, i).
    mul(_dx, dy, w);
}

void LayerFullyConnected::print(tool::TablePrinter &printer, //
                                int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = y();
    auto &filter = w;

    printer.addRow({std::string(depth, '-') + std::to_string(index), //
                    std::string(type()),                             //
                    (std::string)output.dimension(),              //
                    std::to_string(filter.batch()) + ", "            //
                        + std::to_string(filter.channel()),          //
                    std::string("bias=") + (bias() ? "true" : "false")});
}

} // namespace layer
} // namespace deepnet
