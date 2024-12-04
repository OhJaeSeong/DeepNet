/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerActivationSilu.hpp"
#include "deepnet/Debug.hpp"

void gpu_silu(float *y, float *x, size_t n, float *sig_num);

namespace deepnet {
namespace layer {
    
LayerActivationSilu::~LayerActivationSilu() {
    if (sig) {
        delete sig;
        sig = nullptr;
    }
}

void LayerActivationSilu::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    Layer::forward(x);
    sig->forward(x);
    gpu_silu(_y.data(), x.data(), x.size(), (sig->y()).data());
};

void LayerActivationSilu::print(tool::TablePrinter &printer, //
                                     int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = const_cast<TensorGpu &>(y());

    printer.addRow({std::string(depth, '-') + std::to_string(index), //
                    std::string(type()),                             //
                    (std::string)output.dimension(),                 //
                    ""});
}
} // namespace layer
} // namespace deepnet