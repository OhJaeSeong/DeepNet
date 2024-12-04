/// Copyright (c)2021 Electronics and Telecommunications Research
/// Institute(ETRI)
#include <functional>

#include "deepnet/layer/LayerDarknet53.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/layer/LayerActivationLeakyRelu.hpp"
#include "deepnet/layer/LayerBatchNorm.hpp"
#include "deepnet/layer/LayerConvNorm.hpp"
#include "deepnet/layer/LayerConvolutional.hpp"
#include "deepnet/layer/LayerPooling.hpp"
#include "deepnet/layer/LayerSplit.hpp"
#include "deepnet/layer/LayerUpSample.hpp"
#include <tuple>
#include <vector>

namespace deepnet {
namespace layer {

LayerDarknet53A::LayerDarknet53A(const TensorGpu &x, Workspace &workspace,
                                 int filter1, int filter2)
    : LayerSequential(x), _workspace(&workspace), split(nullptr),
      merge(nullptr) {
    DEEPNET_TRACER;

    try {
        *this += split = new LayerSplit(x);
        *this += conv1 = new LayerConvNormLeaky(y(), *_workspace, //
                                                filter1, 1, 1, 1);
        *this += conv2 = new LayerConvNormLeaky(y(), *_workspace, //
                                                filter2, 3, 3, 1, 1);
        *this += merge = new LayerMerge(y());
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void LayerDarknet53A::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);

    x >> split >> conv1 >> conv2;
    merge->forward(conv2->y(), split->y());
}

void LayerDarknet53A::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    Layer::backward(dy);

    dy << merge << conv2 << conv1;
    split->backward(conv1->dx(), merge->dx());
}

LayerDarknet53B::LayerDarknet53B(const TensorGpu &x, Workspace &workspace, //
                                 int filter)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;

    try {
        *this += new LayerConvNormLeaky(x, *_workspace, filter, 3, 3, 2, 1);
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

LayerDarknet53::LayerDarknet53(const TensorGpu &x, Workspace &workspace,
                               bool isDetector, std::function<void(int)> callback)
    : LayerSequential(x), _workspace(&workspace), //
      split36(nullptr), split61(nullptr) {
    DEEPNET_TRACER;

    try {
        if (callback != nullptr) callback(0);
        *this += new LayerConvNormLeaky(x, *_workspace, 32, 3, 3, 1, 1);   // 32
        *this += new LayerConvNormLeaky(y(), *_workspace, 64, 3, 3, 2, 1); // 64
        if (callback != nullptr) callback(2);
        cb = 2;

        std::vector<std::tuple<int, int, int, int>> params = {
            {1, 32, 64, 128},    //
            {2, 64, 128, 256},   //
            {8, 128, 256, 512},  //
            {8, 256, 512, 1024}, //
            {4, 512, 1024, 0},
        };
    
        for (auto param : params) {
            int n, filter1, filter2, filter3;
            std::tie(n, filter1, filter2, filter3) = param;              
            for (auto i = 0; i < n; i++){
                *this += new LayerDarknet53A(y(), *_workspace, //
                                             filter1, filter2);
                cb += 1;
                if (callback != nullptr) callback(cb);
            }

            if (isDetector) {
                if (filter3 == 512)
                    *this += split36 = new LayerSplit(y());

                if (filter3 == 1024)
                    *this += split61 = new LayerSplit(y());
                cb += 1;
                if (callback != nullptr) callback(cb);
            }

            if (filter3 > 0){
                *this += new LayerDarknet53B(y(), *_workspace, filter3);
                cb += 1;
                if (callback != nullptr) callback(cb);
            }      
        }
        cb = 0;
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");

        tool::TablePrinter printer;
        printer.setHeader({"Layer", "Type", "Output", "Filter", "Option"});
        this->print(printer);
        printer.print();

        throw e;
    }
}

} // namespace layer
} // namespace deepnet
