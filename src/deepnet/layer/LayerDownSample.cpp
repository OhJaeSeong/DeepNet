/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerDownSample.hpp"
#include "deepnet/Debug.hpp"
#include <sstream>

namespace deepnet {
namespace layer {

LayerDownSample::LayerDownSample(const TensorGpu &x, int sample)
    : LayerPooling(x, sample, sample, 0, false), _sample(sample) {
    DEEPNET_TRACER;
}

void LayerDownSample::print(tool::TablePrinter &printer, //
                            int depth, int index) const {
    DEEPNET_TRACER;

    auto &output = y();

    printer.addRow({std::string(depth, '-') + std::to_string(index), //
                    std::string(type()),                             //
                    (std::string)output.dimension(),                 //
                    "",                                              //
                    std::string("sample=") + std::to_string(_sample)});
}

} // namespace layer
} // namespace deepnet
