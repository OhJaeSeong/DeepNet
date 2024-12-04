/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerMobileNet.hpp"
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

AddCircle::AddCircle(const TensorGpu &x, Workspace &workspace,
                                 int filter1, int filter2)
    : LayerSequential(x), _workspace(&workspace), split(nullptr),
      merge(nullptr), relu1(nullptr), relu2(nullptr) {
    DEEPNET_TRACER;

    try {
        *this += split = new LayerSplit(x);
        *this += conv1 = new LayerConvolutional(split->y(), *_workspace, filter1, 3, 3, 1, 1, true);
        *this += batch1 = new LayerBatchNorm(conv1->y(), 0.00001);
        *this += relu1 = new LayerActivationRelu(batch1->y());
        
        *this += conv2 = new LayerConvolutional(relu1->y(), *_workspace, filter2, 3, 3, 1, 1, true);
        *this += batch2 = new LayerBatchNorm(conv2->y(), 0.00001);
        *this += merge = new LayerMerge(batch2->y());
        *this += relu2 = new layer::LayerActivationRelu(merge->y());
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void AddCircle::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);

    x >> split >> conv1 >> batch1 >> relu1 >> conv2 >> batch2;
    merge->forward(batch2->y(), split->y());
    merge->y() >> relu2;
}

AddBranch::AddBranch(const TensorGpu &x, Workspace &workspace,
                                 int filter)
    : LayerSequential(x), _workspace(&workspace), split(nullptr),
      merge(nullptr), relu1(nullptr), relu2(nullptr) {
    DEEPNET_TRACER;

    try {
        *this += split = new LayerSplit(x);
        
        *this += conv1 = new LayerConvolutional(split->y(), *_workspace, filter, 3, 3, 2, 1, true);
        *this += batch1 = new LayerBatchNorm(conv1->y(), 0.00001);
        *this += relu1 = new LayerActivationRelu(batch1->y());
        *this += conv2 = new LayerConvolutional(relu1->y(), *_workspace, filter, 3, 3, 1, 1, true);
        *this += batch2 = new LayerBatchNorm(conv2->y(), 0.00001);
        
        *this += avgPool = new layer::LayerPooling(split->y(), 2, 2, 2, 2, 0, 0, false);
        *this += conv3 = new LayerConvolutional(avgPool->y(), *_workspace, filter, 1, 1, 1, 0, true);
        *this += batch3 = new LayerBatchNorm(conv3->y(), 0.00001);

        *this += merge = new LayerMerge(batch3->y());
        *this += relu2 = new layer::LayerActivationRelu(merge->y());
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        throw e;
    }
}

void AddBranch::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);
    x >> split;
    
    x >> conv1 >> batch1 >> relu1 >> conv2 >> batch2;
    split->y() >> avgPool >> conv3 >> batch3;
    merge->forward(batch2->y(), batch3->y());
    merge->y() >> relu2;
}


LayerMobileNet::LayerMobileNet(const TensorGpu &x, Workspace &workspace)
    : LayerSequential(x), _workspace(&workspace)
    {
    DEEPNET_TRACER;

    try {
        *this += conv_0 = new LayerConvolutional(x, *_workspace, 28, 3, 3, 2, 1, true);
        *this += batch_0 = new LayerBatchNorm(y(), 0.00001);
        *this += relu_0 = new LayerActivationRelu(y());
        
        *this += conv_3 = new LayerConvolutional(conv_0->y(), *_workspace, 28, 3, 3, 1, 1, true);
        *this += batch_3 = new LayerBatchNorm(y(), 0.00001);
        *this += relu_3 = new LayerActivationRelu(y());
        
        *this += conv_6 = new LayerConvolutional(conv_3->y(), *_workspace, 56, 3, 3, 1, 1, true);
        *this += batch_6 = new LayerBatchNorm(y(), 0.00001);
        *this += relu_6 = new LayerActivationRelu(y());

        *this += maxPool_9 = new layer::LayerPooling(conv_6->y(), 2, 2, 2, 2, 0, 0, true);

        *this += AddCircle_1 = new AddCircle(maxPool_9->y(), *_workspace, 56, 56); // 10, 13, 16
        *this += AddCircle_2 = new AddCircle(AddCircle_1->y(), *_workspace, 56, 56); // 17, 20, 23
        *this += AddCircle_3 = new AddCircle(AddCircle_2->y(), *_workspace, 56, 56); // 24, 27, 30
        
        *this += AddBranch_1 = new AddBranch(AddCircle_3->y(), *_workspace, 88); // 31, 34, 36, 37, 40

        *this += AddCircle_4 = new AddCircle(AddBranch_1->y(), *_workspace, 88, 88); // 41, 44, 47
        *this += AddCircle_5 = new AddCircle(AddCircle_4->y(), *_workspace, 88, 88); // 48, 51, 54
        *this += AddCircle_6 = new AddCircle(AddCircle_5->y(), *_workspace, 88, 88); // 55, 58, 61 1차 저장

        *this += AddBranch_2 = new AddBranch(AddCircle_6->y(), *_workspace, 88); // 62, 65, 67, 68, 71

        *this += AddCircle_7 = new AddCircle(AddBranch_2->y(), *_workspace, 88, 88); // 72, 75, 78 2차 저장

        *this += AddBranch_3 = new AddBranch(AddCircle_7->y(), *_workspace, 224); // 79, 82, 84, 85, 88

        *this += AddCircle_8 = new AddCircle(AddBranch_3->y(), *_workspace, 224, 224); // 89, 92, 95
        *this += AddCircle_9 = new AddCircle(AddCircle_8->y(), *_workspace, 224, 224); // 96, 99, 102

        // FPN
        *this += conv_103 = new LayerConvolutional(AddCircle_6->y(), *_workspace, 56, 1, 1, 1, 0, true); // x1
        *this += batch_103 = new LayerBatchNorm(conv_103->y(), 0.00001);
        *this += conv_104 = new LayerConvolutional(AddCircle_7->y(), *_workspace, 56, 1, 1, 1, 0, true); // x2
        *this += batch_104 = new LayerBatchNorm(conv_104->y(), 0.00001);
        *this += conv_105 = new LayerConvolutional(AddCircle_9->y(), *_workspace, 56, 1, 1, 1, 0, true); // x3
        *this += batch_105 = new LayerBatchNorm(conv_105->y(), 0.00001);
  
        *this += merge3and2 = new layer::LayerMerge(batch_104->y());
        *this += merge2and1 = new layer::LayerMerge(batch_103->y());

        *this += upsample3to2 = new LayerUpSample(batch_105->y(), 2); 
        *this += upsample2to1 = new LayerUpSample(merge3and2->y(), 2);// merge2and1(=x1), merge3and2(=x2), conv_105(=x3)
        
        *this += conv_146 = new LayerConvolutional(merge2and1->y(), *_workspace, 56, 3, 3, 1, 1, true); // x1
        *this += batch_146 = new LayerBatchNorm(conv_146->y(), 0.00001);
        
        *this += conv_147 = new LayerConvolutional(merge3and2->y(), *_workspace, 56, 3, 3, 1, 1, true); // x2
        *this += batch_147 = new LayerBatchNorm(conv_147->y(), 0.00001);

        *this += conv_148 = new LayerConvolutional(batch_105->y(), *_workspace, 56, 3, 3, 1, 1, true); // x3
        *this += batch_148 = new LayerBatchNorm(conv_148->y(), 0.00001);

        *this += add_150 = new layer::LayerMerge(batch_147->y());
        *this += add_152 = new layer::LayerMerge(batch_148->y());

        *this += conv_149 = new LayerConvolutional(batch_146->y(), *_workspace, 56, 3, 3, 2, 1, true); // x2_add
        *this += batch_149 = new LayerBatchNorm(conv_149->y(), 0.00001);

        *this += conv_151 = new LayerConvolutional(add_150->y(), *_workspace, 56, 3, 3, 2, 1, true); // x3_add
        *this += batch_151 = new LayerBatchNorm(conv_151->y(), 0.00001);
        
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");

        tool::TablePrinter printer;
        printer.setHeader({"Layer", "Type", "Output", "Filter", "Option"});
        this->print(printer);
        printer.print();

        throw e;
    }
}
void LayerMobileNet::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    // Layer::forward(x);
    x >> conv_0 >> batch_0 >> relu_0 >> conv_3 >> batch_3 >> relu_3 >> conv_6 >> batch_6 >> relu_6;
    relu_6->y() >> maxPool_9 >> AddCircle_1 >> AddCircle_2 >> AddCircle_3 >> AddBranch_1 >> AddCircle_4;
    AddCircle_4->y() >> AddCircle_5 >> AddCircle_6 >> AddBranch_2 >> AddCircle_7 >> AddBranch_3 >> AddCircle_8 >> AddCircle_9;
    
    AddCircle_6->y() >> conv_103 >> batch_103;
    AddCircle_7->y() >> conv_104 >> batch_104;
    AddCircle_9->y() >> conv_105 >> batch_105;

    batch_105->y() >> upsample3to2;
    merge3and2->forward(batch_104->y(), upsample3to2->y()); //x2 = merge3and2, x3 = conv_105
    merge3and2->y() >> upsample2to1;
    merge2and1->forward(batch_103->y(), upsample2to1->y()); //x1 = merge2and1, add_145

    merge2and1->y() >> conv_146 >> batch_146 >> conv_149 >> batch_149; //146->y : x1
    merge3and2->y() >> conv_147 >> batch_147;
    batch_105->y() >> conv_148 >> batch_148;
    
    add_150->forward(batch_147->y(), batch_149->y()); //x2
    add_150->y() >> conv_151 >> batch_151;
    add_152->forward(batch_148->y(), batch_151->y()); //x3
}

} // namespace layer
} // namespace deepnet