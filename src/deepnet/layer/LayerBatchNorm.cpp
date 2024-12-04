/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerBatchNorm.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/Weight.hpp"
#include <sstream>

namespace deepnet {
namespace layer {

LayerBatchNorm::LayerBatchNorm(const TensorGpu &x, double epsilon)
    : LayerWeighted(x), _epsilon(epsilon), _n(0.0),
      _bnScaleBiasMeanVarDesc(nullptr) {
    DEEPNET_ASSERT(!x.isEmpty());
    DEEPNET_ASSERT(epsilon >= CUDNN_BN_MIN_EPSILON);

    auto dims = x.dimension();

    _y.setDimension(dims);

    dims = Dimension(1, dims.channel(), 1, 1);

    w.setDimension(dims);
    b.setDimension(dims);

    mean.setDimension(dims);
    variance.setDimension(dims);

    randomizeWeight(Weight::InitMethod::Xavier);

    SAFE_CUDNN(cudnnCreateTensorDescriptor(&_bnScaleBiasMeanVarDesc));
    SAFE_CUDNN(cudnnDeriveBNTensorDescriptor(
        _bnScaleBiasMeanVarDesc, x.descriptor(),
        cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL));
}

LayerBatchNorm::~LayerBatchNorm() {
    if (_bnScaleBiasMeanVarDesc) {
        UNSAFE_CUDNN(cudnnDestroyTensorDescriptor(_bnScaleBiasMeanVarDesc));
        _bnScaleBiasMeanVarDesc = nullptr;
    }
}

void LayerBatchNorm::train(void) {
    LayerWeighted::train();

    auto dims = w.dimension();

    dw.setDimension(dims);
    db.setDimension(dims);
    _resultSaveMean.setDimension(dims);
    _resultSaveInvVariance.setDimension(dims);
}

void LayerBatchNorm::eval(void) {
    LayerWeighted::eval();

    dw.setDimension(0, 0, 0, 0);
    db.setDimension(0, 0, 0, 0);
    _resultSaveMean.setDimension(0, 0, 0, 0);
    _resultSaveInvVariance.setDimension(0, 0, 0, 0);
}

void LayerBatchNorm::randomizeWeight(Weight::InitMethod method) {
    auto dimension = w.dimension();
    TensorCpu temp(dimension);

    if (method == Weight::InitMethod::Fill01) {
        temp.fill(0.1f);
        b.from(temp);
        mean.from(temp);
        w.from(temp);
        variance.from(temp);
    } else {
        temp.fill(0.0f);
        b.from(temp);
        mean.from(temp);

        temp.fill(1.0f);
        w.from(temp);
        variance.from(temp);
    }
}

void LayerBatchNorm::readWeight(FILE *file, Weight::Format format) {
    LayerWeighted::readWeight(file, format);

    Weight::readWeight(file, mean, format);
    Weight::readWeight(file, variance, format);
}

void LayerBatchNorm::writeWeight(FILE *file, Weight::Format format) const {
    LayerWeighted::writeWeight(file, format);

    Weight::writeWeight(file, mean, format);
    Weight::writeWeight(file, variance, format);
}

void LayerBatchNorm::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    LayerWeighted::forward(x);

    float alpha = 1.0f;
    float beta = 0.0f;

    if (_training) {
        // 학습 시에만 N 값을 증가시킨다.
        _n++;

        SAFE_CUDNN(cudnnBatchNormalizationForwardTraining( //
            cudnn_handle,                                  //
            cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL, //
            &alpha,                                        //
            &beta,                                         //
            x.descriptor(),                                //
            x.data(),                                      //
            _y.descriptor(),                               //
            _y.data(),                                     //
            _bnScaleBiasMeanVarDesc,                       //
            w.data(),                                      // bnScale
            b.data(),                                      // bnBias
            1.0 / _n,                        // exponentialAverageFactor
            mean.data(),                     // resultRunningMean
            variance.data(),                 // resultRunningVariance
            _epsilon,                        //
            _resultSaveMean.data(),          // resultSaveMean
            _resultSaveInvVariance.data())); // resultSaveInvVariance
    } else {
        SAFE_CUDNN(cudnnBatchNormalizationForwardInference( //
            cudnn_handle,                                   //
            cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,  //
            &alpha,                                         //
            &beta,                                          //
            x.descriptor(),                                 //
            x.data(),                                       //
            _y.descriptor(),                                //
            _y.data(),                                      //
            _bnScaleBiasMeanVarDesc,                        //
            w.data(),                                       // bnScale
            b.data(),                                       // bnBias
            mean.data(),                                    // resultRunningMean
            variance.data(), // resultRunningVariance
            _epsilon));
    }
}

void LayerBatchNorm::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    LayerWeighted::backward(dy);

    float alpha = 1.0f;
    float beta = 0.0f;

    SAFE_CUDNN(cudnnBatchNormalizationBackward(        //
        cudnn_handle,                                  //
        cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL, //
        &alpha,                                        //
        &beta,                                         //
        &alpha,                                        //
        &beta,                                         //
        _px->descriptor(),                             //
        _px->data(),                                   //
        dy.descriptor(),                               //
        dy.data(),                                     //
        _dx.descriptor(),                              //
        _dx.data(),                                    //
        _bnScaleBiasMeanVarDesc,                       //
        w.data(),                                      //
        dw.data(),                                     //
        db.data(),                                     //
        _epsilon,                                      //
        _resultSaveMean.data(),                        //
        _resultSaveInvVariance.data()));
}

void LayerBatchNorm::print(tool::TablePrinter &printer, int depth,
                           int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());

    printer.addRow({std::string(depth, '-') + std::to_string(index), //
                    std::string(type()),                             //
                    (std::string)output.dimension(),                 //
                    "",                                              //
                    std::string("epsilon=") + std::to_string(_epsilon)});
}

#define SS(x) std::to_string(x)

void LayerBatchNorm::printWeight(tool::TablePrinter &printer, int depth,
                                 int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());

    TensorCpu scale(w);
    TensorCpu bias(b);
    TensorCpu u(mean);
    TensorCpu v(variance);

    float scale_min = 0.0f, scale_max = 0.0f;
    std::tie(scale_min, scale_max) = scale.getMinMax();

    float bias_min = 0.0f, bias_max = 0.0f;
    std::tie(bias_min, bias_max) = bias.getMinMax();

    float mean_min = 0.0f, mean_max = 0.0f;
    std::tie(mean_min, mean_max) = u.getMinMax();

    float variance_min = 0.0f, variance_max = 0.0f;
    std::tie(variance_min, variance_max) = v.getMinMax();

    printer.addRow({std::string(depth, '-') + std::to_string(index),
                    std::string(type()), //
                    (std::string)output.dimension(),
                    std::string("s=") + SS(scale_min) + "~" + SS(scale_max) + //
                        ", b=" + SS(bias_min) + "~" + SS(bias_max) +          //
                        ", u=" + SS(mean_min) + "~" + SS(mean_max) +          //
                        ", v=" + SS(variance_min) + "~" + SS(variance_max)});
}

unsigned char LayerBatchNorm::checksum(void) const {
    auto checksum_w = TensorCpu(w).checksum();
    auto checksum_b = TensorCpu(b).checksum();
    auto checksum_mean = TensorCpu(mean).checksum();
    auto checksum_variance = TensorCpu(variance).checksum();

    return ~(~checksum_w + ~checksum_b + ~checksum_mean + ~checksum_variance);
}

} // namespace layer
} // namespace deepnet
