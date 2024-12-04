/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerHBResNet50.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/layer/LayerActivationRelu.hpp"
#include "deepnet/layer/LayerBatchNorm.hpp"
#include "deepnet/layer/LayerConvolutional.hpp"
#include "deepnet/layer/LayerPooling.hpp"
#include "deepnet/layer/LayerSplit.hpp"
#include "deepnet/layer/LayerUpSample.hpp"
#include "deepnet/layer/LayerActivationRelu.hpp"
#include <tuple>
#include <vector>
#include <iostream>

namespace deepnet {
namespace layer {

BottleneckDLoop::BottleneckDLoop(const TensorGpu &x, Workspace &workspace,
                                 int FrontFilter, int BackFilter, bool downsample, int stride)
    : LayerSequential(x), _workspace(&workspace), split(nullptr),
      merge(nullptr), relu1(nullptr), relu2(nullptr), relu3(nullptr) {
    DEEPNET_TRACER;

    try {
        *this += split = new LayerSplit(x);
        *this += conv1 = new LayerConvolutional(split->y(), *_workspace, FrontFilter, 1, 1, 1, 0, false);
        *this += batch1 = new LayerBatchNorm(conv1->y(), 0.00001);
        *this += relu1 = new LayerActivationRelu(batch1->y());
        
        *this += conv2 = new LayerConvolutional(relu1->y(), *_workspace, FrontFilter, 3, 3, stride, 1, false);
        *this += batch2 = new LayerBatchNorm(conv2->y(), 0.00001);
        *this += relu2 = new layer::LayerActivationRelu(batch2->y());

        *this += conv3 = new LayerConvolutional(relu2->y(), *_workspace, BackFilter, 1, 1, 1, 0, false);
        *this += batch3 = new LayerBatchNorm(conv3->y(), 0.00001);

        if (downsample) {
            *this += conv_down = new layer::LayerConvolutional( //
                split->y(), workspace, conv3->y().channel(), 1, 1, stride, 0, false);
            
            *this += batch_down = new LayerBatchNorm(conv_down->y(), 0.00001);
        }

        *this += merge = new LayerMerge(batch3->y());
        *this += relu3 = new layer::LayerActivationRelu(merge->y());

        
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        throw e;
    }
}

void BottleneckDLoop::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);

   x >> split
        >> conv1 >> batch1 >> relu1 //
        >> conv2 >> batch2 >> relu2 //
        >> conv3 >> batch3;

    
    auto *identity = &split->y();

    if (conv_down)
        identity = &((*identity) >> conv_down >> batch_down);
    
    merge->forward(batch3->y(), *identity);
    merge->y() >> relu3;
}


BottleneckLoop::BottleneckLoop(const TensorGpu &x, Workspace &workspace,
                                 int FrontFilter, int BackFilter)
    : LayerSequential(x), _workspace(&workspace), split(nullptr),
      merge(nullptr), relu1(nullptr), relu2(nullptr), relu3(nullptr) {
    DEEPNET_TRACER;

    try {
        *this += split = new LayerSplit(x);
        *this += conv1 = new LayerConvolutional(split->y(), *_workspace, FrontFilter, 1, 1, 1, 0, false);
        *this += batch1 = new LayerBatchNorm(conv1->y(), 0.00001);
        *this += relu1 = new LayerActivationRelu(batch1->y());
        
        *this += conv2 = new LayerConvolutional(relu1->y(), *_workspace, FrontFilter, 3, 3, 1, 1, false);
        *this += batch2 = new LayerBatchNorm(conv2->y(), 0.00001);
        *this += relu2 = new layer::LayerActivationRelu(batch2->y());

        *this += conv3 = new LayerConvolutional(relu2->y(), *_workspace, BackFilter, 1, 1, 1, 0, false);
        *this += batch3 = new LayerBatchNorm(conv3->y(), 0.00001);

        *this += merge = new LayerMerge(batch3->y());
        *this += relu3 = new layer::LayerActivationRelu(merge->y());

        
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        throw e;
    }
}

void BottleneckLoop::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);

   x >> split
        >> conv1 >> batch1 >> relu1 //
        >> conv2 >> batch2 >> relu2 //
        >> conv3 >> batch3;
    
    merge->forward(batch3->y(), split->y());
    merge->y() >> relu3;
}


LayerHBResNet50::LayerHBResNet50(const TensorGpu &x, Workspace &workspace)
    : LayerSequential(x), _workspace(&workspace)
    {
    DEEPNET_TRACER;

    try {
        *this += conv_0 = new LayerConvolutional(x, *_workspace, 64, 7, 7, 2, 3, false);
        *this += batch_0 = new LayerBatchNorm(y(), 0.00001);
        *this += relu_0 = new LayerActivationRelu(y());
        
        *this += maxPool_2 = new layer::LayerPooling(conv_0->y(), 3, 3, 2, 2, 1, 1, true);

        *this += BottleneckLoop1 = new BottleneckDLoop(maxPool_2->y(), *_workspace, 64, 256, true, 1);
        *this += BottleneckLoop2 = new BottleneckLoop(BottleneckLoop1->y(), *_workspace, 64, 256);
        *this += BottleneckLoop3 = new BottleneckLoop(BottleneckLoop2->y(), *_workspace, 64, 256);
        
        *this += BottleneckLoop4 = new BottleneckDLoop(BottleneckLoop3->y(), *_workspace, 128, 512, true, 2);
        *this += BottleneckLoop5 = new BottleneckLoop(BottleneckLoop4->y(), *_workspace, 128, 512);
        *this += BottleneckLoop6 = new BottleneckLoop(BottleneckLoop5->y(), *_workspace, 128, 512);
        *this += BottleneckLoop7 = new BottleneckLoop(BottleneckLoop6->y(), *_workspace, 128, 512); // relu53 분기

        *this += BottleneckLoop8 = new BottleneckDLoop(BottleneckLoop7->y(), *_workspace, 256, 1024, true, 2);
        *this += BottleneckLoop9 = new BottleneckLoop(BottleneckLoop8->y(), *_workspace, 256, 1024);
        *this += BottleneckLoop10 = new BottleneckLoop(BottleneckLoop9->y(), *_workspace, 256, 1024);
        *this += BottleneckLoop11 = new BottleneckLoop(BottleneckLoop10->y(), *_workspace, 256, 1024);
        *this += BottleneckLoop12 = new BottleneckLoop(BottleneckLoop11->y(), *_workspace, 256, 1024); 
        *this += BottleneckLoop13 = new BottleneckLoop(BottleneckLoop12->y(), *_workspace, 256, 1024); // relu96 분기

        *this += BottleneckLoop14 = new BottleneckDLoop(BottleneckLoop13->y(), *_workspace, 512, 2048, true, 2);
        *this += BottleneckLoop15 = new BottleneckLoop(BottleneckLoop14->y(), *_workspace, 512, 2048);
        *this += BottleneckLoop16 = new BottleneckLoop(BottleneckLoop15->y(), *_workspace, 512, 2048);
        
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");

        tool::TablePrinter printer;
        printer.setHeader({"Layer", "Type", "Output", "Filter", "Option"});
        this->print(printer);
        printer.print();

        throw e;
    }
}
void LayerHBResNet50::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    // Layer::forward(x);
    x >> conv_0 >> batch_0 >> relu_0 >> maxPool_2;
    maxPool_2->y() >> BottleneckLoop1 >> BottleneckLoop2 >> BottleneckLoop3;
    BottleneckLoop3->y() >> BottleneckLoop4
                         >> BottleneckLoop5 >> BottleneckLoop6 >> BottleneckLoop7;
    BottleneckLoop7->y() >> BottleneckLoop8
                         >> BottleneckLoop9 >> BottleneckLoop10 >> BottleneckLoop11 >> BottleneckLoop12 >> BottleneckLoop13;
    BottleneckLoop13->y() >> BottleneckLoop14
                          >> BottleneckLoop15 >> BottleneckLoop16;
}
} // namespace layer
} // namespace deepnet