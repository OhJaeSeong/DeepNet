/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerYolov8n.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/layer/LayerActivationSilu.hpp"
#include "deepnet/layer/LayerBatchNorm.hpp"
#include "deepnet/layer/LayerConvolutional.hpp"
#include "deepnet/layer/LayerPooling.hpp"
#include "deepnet/layer/LayerSplit.hpp"
#include "deepnet/layer/LayerUpSample.hpp"
#include "deepnet/layer/LayerChunk.hpp"
#include <tuple>
#include <math.h>
#include <vector>
#include <algorithm>
#include <iostream>

namespace deepnet {
namespace layer {

C2f2::C2f2(const TensorGpu &x, Workspace &workspace, int input_filter, bool add = true)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;
    try {
        this->add = add;
        *this += conv_first = new LayerConvolutional(x, *_workspace, input_filter * 2, 1, 1, 1, 0, true);
        *this += act_first = new LayerActivationSilu(conv_first->y());
        
        *this += chunk1 = new LayerChunk(act_first->y(), 2, 1, 0); // number dim count
        *this += chunk2 = new LayerChunk(act_first->y(), 2, 1, 1);
        
        *this += cat1 = new LayerConcat(chunk1->y(), chunk1->y());
        *this += cat2 = new LayerConcat(cat1->y(), chunk1->y());
        *this += cat3 = new LayerConcat(cat2->y(), chunk1->y());

        *this += conv_final = new LayerConvolutional(cat3->y(), *_workspace, input_filter * 2, 1, 1, 1, 0, true);
        *this += act_final = new LayerActivationSilu(conv_final->y());

        *this += conv0 = new LayerConvolutional(chunk2->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act0 = new LayerActivationSilu(conv0->y());
        *this += conv1 = new LayerConvolutional(act0->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act1 = new LayerActivationSilu(conv1->y());

        if(add){
            *this += add1 = new LayerCalculateEach(chunk2->y(), act1->y(), 0);
        }            

        *this += conv2 = new LayerConvolutional(act1->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act2 = new LayerActivationSilu(conv2->y());
        *this += conv3 = new LayerConvolutional(act2->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act3 = new LayerActivationSilu(conv3->y());

        if(add){
            *this += add2 = new LayerCalculateEach(add1->y(), act3->y(), 0);
        }

    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void C2f2::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    Layer::forward(x);
    x >> conv_first >> act_first;
    act_first->y() >> chunk1;
    act_first->y() >> chunk2;  

    chunk2->y() >> conv0 >> act0 >> conv1 >> act1;

    if(this->add){
        add1->forward(chunk2->y(), act1->y());
        add1->y() >> conv2 >> act2 >> conv3 >> act3;
        add2->forward(add1->y(), act3->y());

        cat1->forward(chunk1->y(), chunk2->y());
        cat2->forward(cat1->y(), add1->y());
        cat3->forward(cat2->y(), add2->y());
        cat3->y() >> conv_final >> act_final;
    }else{
        act1->y() >> conv2 >> act2 >> conv3 >> act3;
        cat1->forward(chunk1->y(), chunk2->y());
        cat2->forward(cat1->y(), act1->y());
        cat3->forward(cat2->y(), act3->y());
        cat3->y() >> conv_final >> act_final;
    }
}

C2f4::C2f4(const TensorGpu &x, Workspace &workspace, int input_filter, bool add = true)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;
    try {
        *this += conv_first = new LayerConvolutional(x, *_workspace, input_filter * 2, 1, 1, 1, 0, true);
        *this += act_first = new LayerActivationSilu(conv_first->y());
        
        *this += chunk1 = new LayerChunk(act_first->y(), 2, 1, 0); // number dim count
        *this += chunk2 = new LayerChunk(act_first->y(), 2, 1, 1);
        
        *this += cat1 = new LayerConcat(chunk1->y(), chunk1->y());
        *this += cat2 = new LayerConcat(cat1->y(), chunk1->y());
        *this += cat3 = new LayerConcat(cat2->y(), chunk1->y());
        *this += cat4 = new LayerConcat(cat3->y(), chunk1->y());
        *this += cat5 = new LayerConcat(cat4->y(), chunk1->y());

        *this += conv_final = new LayerConvolutional(cat5->y(), *_workspace, input_filter * 2, 1, 1, 1, 0, true);
        *this += act_final = new LayerActivationSilu(conv_final->y());

        *this += conv0 = new LayerConvolutional(chunk2->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act0 = new LayerActivationSilu(conv0->y());
        *this += conv1 = new LayerConvolutional(act0->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act1 = new LayerActivationSilu(conv1->y());
        *this += add1 = new LayerCalculateEach(chunk2->y(), act1->y(), 0);

        *this += conv2 = new LayerConvolutional(add1->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act2 = new LayerActivationSilu(conv2->y());
        *this += conv3 = new LayerConvolutional(act2->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act3 = new LayerActivationSilu(conv3->y());
        *this += add2 = new LayerCalculateEach(add1->y(), act3->y(), 0);


        *this += conv4 = new LayerConvolutional(add2->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act4 = new LayerActivationSilu(conv4->y());
        *this += conv5 = new LayerConvolutional(act4->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act5 = new LayerActivationSilu(conv5->y());
        *this += add3 = new LayerCalculateEach(add2->y(), act3->y(), 0);

        *this += conv6 = new LayerConvolutional(add3->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act6 = new LayerActivationSilu(conv6->y());
        *this += conv7 = new LayerConvolutional(act6->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act7 = new LayerActivationSilu(conv7->y());
        *this += add4 = new LayerCalculateEach(add3->y(), act3->y(), 0);


    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void C2f4::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    Layer::forward(x);
    x >> conv_first >> act_first;
    act_first->y() >> chunk1;
    act_first->y() >> chunk2;  

    chunk2->y() >> conv0 >> act0 >> conv1 >> act1;
    add1->forward(chunk2->y(), act1->y());
    add1->y() >> conv2 >> act2 >> conv3 >> act3;
    add2->forward(add1->y(), act3->y());
    add2->y() >> conv4 >> act4 >> conv5 >> act5;
    add3->forward(add2->y(), act5->y());
    add3->y() >> conv6 >> act6 >> conv7 >> act7;
    add4->forward(add3->y(), act7->y());

    cat1->forward(chunk1->y(), chunk2->y());
    cat2->forward(cat1->y(), add1->y());
    cat3->forward(cat2->y(), add2->y());
    cat4->forward(cat3->y(), add3->y());
    cat5->forward(cat4->y(), add4->y());
    cat5->y() >> conv_final >> act_final;
}

LayerYolov8n::LayerYolov8n(const TensorGpu &x, Workspace &workspace, std::function<void(int)> callback)
    : LayerSequential(x), _workspace(&workspace)
    {
    DEEPNET_TRACER;

    try {
        if (callback != nullptr) callback(0);
        *this += conv_0 = new LayerConvolutional(x, *_workspace, 48, 3, 3, 2, 1, true);
        *this += act_0 = new LayerActivationSilu(conv_0->y());
        
        *this += conv_1 = new LayerConvolutional(act_0->y(), *_workspace, 96, 3, 3, 2, 1, true);
        *this += act_1 = new LayerActivationSilu(conv_1->y());
        *this += c2f2_1 = new C2f2(act_1->y(), *_workspace, 48);

        *this += conv_2 = new LayerConvolutional(c2f2_1->result(), *_workspace, 192, 3, 3, 2, 1, true);
        *this += act_2 = new LayerActivationSilu(conv_2->y());
        *this += c2f4_1 = new C2f4(act_2->y(), *_workspace, 96); // branch /

        *this += conv_3 = new LayerConvolutional(c2f4_1->result(), *_workspace, 384, 3, 3, 2, 1, true);
        *this += act_3 = new LayerActivationSilu(conv_3->y());
        *this += c2f4_2 = new C2f4(act_3->y(), *_workspace, 192); // branch /

        *this += conv_4 = new LayerConvolutional(c2f4_2->result(), *_workspace, 576, 3, 3, 2, 1, true);
        *this += act_4 = new LayerActivationSilu(conv_4->y());
        *this += c2f2_2 = new C2f2(act_4->y(), *_workspace, 288);

        // SPPF
        *this += sppf_conv_1 = new LayerConvolutional(c2f2_2->result(), *_workspace, 288, 1, 1, 1, 0, true);
        *this += sppf_act_1 = new LayerActivationSilu(sppf_conv_1->y());
        *this += maxpool1 = new LayerPooling(sppf_act_1->y() ,5, 1, 2, true);
        *this += maxpool2 = new LayerPooling(maxpool1->y() ,5, 1, 2, true);
        *this += maxpool3 = new LayerPooling(maxpool2->y() ,5, 1, 2, true);

        *this += sppf_cat1 = new LayerConcat(sppf_act_1->y(), maxpool1->y());
        *this += sppf_cat2 = new LayerConcat(sppf_cat1->y(), maxpool2->y());
        *this += sppf_cat3 = new LayerConcat(sppf_cat2->y(), maxpool3->y());
        *this += sppf_conv_2 = new LayerConvolutional(sppf_cat3->y(), *_workspace, 576, 1, 1, 1, 0, true);
        *this += sppf_act_2 = new LayerActivationSilu(sppf_conv_2->y()); // branch /
        //

        *this += up1 = new LayerUpSample(sppf_act_2->y(), 2);
        *this += branch_merge1 = new LayerConcat(up1->y(), c2f4_2->result());
        *this += c2f2_3 = new C2f2(branch_merge1->y(), *_workspace, 192, false); // branch /
        
        *this += up2 = new LayerUpSample(c2f2_3->result(), 2);
        *this += branch_merge2 = new LayerConcat(up2->y(), c2f4_1->result());
        *this += c2f2_4 = new C2f2(branch_merge2->y(), *_workspace, 96, false); // 15

        *this += conv_5 = new LayerConvolutional(c2f2_4->result(), *_workspace, 192, 3, 3, 2, 1, true); // 16
        *this += act_5 = new LayerActivationSilu(conv_5->y());
        *this += branch_merge3 = new LayerConcat(act_5->y(), c2f2_3->result());
        *this += c2f2_5 = new C2f2(branch_merge3->y(), *_workspace, 192, false); // 18

        *this += conv_6 = new LayerConvolutional(c2f2_5->result(), *_workspace, 384, 3, 3, 2, 1, true); // 19
        *this += act_6 = new LayerActivationSilu(conv_6->y());
        *this += branch_merge4 = new LayerConcat(act_6->y(), sppf_act_2->y());
        *this += c2f2_6 = new C2f2(branch_merge4->y(), *_workspace, 288, false); // 21

    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");

        tool::TablePrinter printer;
        printer.setHeader({"Layer", "Type", "Output", "Filter", "Option"});
        this->print(printer);
        printer.print();

        throw e;
    }
}

void LayerYolov8n::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    Layer::forward(x);

    x >> conv_0 >> act_0 >> conv_1 >> act_1 >> c2f2_1;
    c2f2_1->result() >> conv_2 >> act_2 >> c2f4_1;
    c2f4_1->result() >> conv_3 >> act_3 >> c2f4_2;
    c2f4_2->result() >> conv_4 >> act_4 >> c2f2_2;

    c2f2_2->result() >> sppf_conv_1 >> sppf_act_1 >> maxpool1 >> maxpool2 >> maxpool3;
    sppf_cat1->forward(sppf_act_1->y(), maxpool1->y());
    sppf_cat2->forward(sppf_cat1->y(), maxpool2->y());
    sppf_cat3->forward(sppf_cat2->y(), maxpool3->y());
    sppf_cat3->y() >> sppf_conv_2 >> sppf_act_2;
    
    sppf_act_2->y() >> up1;
    branch_merge1->forward(up1->y(), c2f4_2->result());
    branch_merge1->y() >> c2f2_3;
    
    c2f2_3->result() >> up2;
    branch_merge2->forward(up2->y(), c2f4_1->result());
    branch_merge2->y() >> c2f2_4;

    c2f2_4->result() >> conv_5 >> act_5;
    branch_merge3->forward(act_5->y(), c2f2_3->result());
    branch_merge3->y() >> c2f2_5;

    c2f2_5->result() >> conv_6 >> act_6;
    branch_merge4->forward(act_6->y(), sppf_act_2->y());
    branch_merge4->y() >> c2f2_6;
}
} // namespace layer
} // namespace deepnet