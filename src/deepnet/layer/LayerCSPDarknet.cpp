/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerCSPDarknet.hpp"

#include <math.h>

#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>

#include "deepnet/Debug.hpp"
#include "deepnet/layer/LayerActivationRelu.hpp"
#include "deepnet/layer/LayerActivationSilu.hpp"
#include "deepnet/layer/LayerBatchNorm.hpp"
#include "deepnet/layer/LayerConvNorm.hpp"
#include "deepnet/layer/LayerConvolutional.hpp"
#include "deepnet/layer/LayerPooling.hpp"
#include "deepnet/layer/LayerSplit.hpp"
#include "deepnet/layer/LayerUpSample.hpp"

namespace deepnet {
namespace layer {

BottleneckList::BottleneckList(const TensorGpu &x, Workspace &workspace, int filter, bool shortcut)
    : LayerSequential(x), _workspace(&workspace), split1(nullptr), split2(nullptr), merge1(nullptr), merge2(nullptr) {
    DEEPNET_TRACER;

    try {
        if(shortcut == true){
            this->addcut = true;
            *this += split1 = new LayerSplit(x);
            *this += conv1 = new LayerConvolutional(split1->y(), *_workspace, filter, 1, 1, 1, 0, false);
            *this += batch1 = new LayerBatchNorm(conv1->y(), 0.001);
            *this += act1 = new LayerActivationSilu(batch1->y());
        }else{
            this->addcut = false;
            *this += conv1 = new LayerConvolutional(x, *_workspace, filter, 1, 1, 1, 0, false);
            *this += batch1 = new LayerBatchNorm(conv1->y(), 0.001);
            *this += act1 = new LayerActivationSilu(batch1->y());
        }

        *this += conv2 = new LayerConvolutional(act1->y(), *_workspace, filter, 3, 3, 1, 1, false);
        *this += batch2 = new LayerBatchNorm(conv2->y(), 0.001);
        *this += act2 = new LayerActivationSilu(batch2->y());
        
        if(shortcut == true){
            *this += merge1 = new LayerMerge(act2->y());
            *this += split2 = new LayerSplit(merge1->y());
            
            *this += conv3 = new LayerConvolutional(split2->y(), *_workspace, filter, 1, 1, 1, 0, false);
            *this += batch3 = new LayerBatchNorm(conv3->y(), 0.001);
            *this += act3 = new LayerActivationSilu(batch3->y());
        }else{
            *this += conv3 = new LayerConvolutional(conv2->y(), *_workspace, filter, 1, 1, 1, 0, false);
            *this += batch3 = new LayerBatchNorm(conv3->y(), 0.001);
            *this += act3 = new LayerActivationSilu(batch3->y());
        }

        *this += conv4 = new LayerConvolutional(conv3->y(), *_workspace, filter, 3, 3, 1, 1, false);
        *this += batch4 = new LayerBatchNorm(conv4->y(), 0.001);
        *this += act4 = new LayerActivationSilu(batch4->y());
        if(shortcut == true){
            *this += merge2 = new LayerMerge(act4->y());
        }

    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void BottleneckList::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);
    if(this->addcut){
        x >> split1 >> conv1 >> batch1 >> act1 >> conv2 >> batch2 >> act2;
        merge1->forward(act2->y(), split1->y());
        merge1->y() >> split2 >> conv3 >> batch3 >> act3 >> conv4 >> batch4 >> act4;
        merge2->forward(act4->y(), split2->y());

    }else{
        x >> conv1 >> batch1 >> act1 >> conv2 >> batch2 >> act2 >> conv3 >> batch3 >> act3 >> conv4 >> batch4 >> act4;
    }
    
}

CSPLayer::CSPLayer(const TensorGpu &x, Workspace &workspace, int start_filter, int type, bool shortcut)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;

    try {
        *this += start_split = new LayerSplit(x);
        *this += conv_middle1 = new LayerConvolutional(x, *_workspace, start_filter/2, 1, 1, 1, 0, false);
        *this += batch_middle1 = new LayerBatchNorm(conv_middle1->y(), 0.001);
        *this += act_middle1 = new LayerActivationSilu(batch_middle1->y());

        *this += conv_middle2 = new LayerConvolutional(x, *_workspace, start_filter/2, 1, 1, 1, 0, false);
        *this += batch_middle2 = new LayerBatchNorm(conv_middle2->y(), 0.001);
        *this += act_middle2 = new LayerActivationSilu(batch_middle2->y());
        
        *this += cat = new LayerConcat(act_middle1->y(), act_middle2->y());
        
        *this += conv_end = new LayerConvolutional(cat->y(), *_workspace, start_filter, 1, 1, 1, 0, false);
        *this += batch_end = new LayerBatchNorm(conv_end->y(), 0.001);
        *this += act_end = new LayerActivationSilu(batch_end->y());
        
        *this += bottleneck1 = new BottleneckList(act_middle1->y(), *_workspace, start_filter/2, shortcut);
        this->num_type = 1;
        
        if(type == 3){
            this->num_type = 3;
            *this += bottleneck2 = new BottleneckList(bottleneck1->y(), *_workspace, start_filter/2, shortcut);
            *this += bottleneck3 = new BottleneckList(bottleneck2->y(), *_workspace, start_filter/2, shortcut);
        }else if(type == 2){
            this->num_type = 2;
            *this += bottleneck2 = new BottleneckList(bottleneck1->y(), *_workspace, start_filter/2, shortcut);
        }
        
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void CSPLayer::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    Layer::forward(x);
    x >> start_split >> conv_middle1 >> batch_middle1 >> act_middle1 >> bottleneck1;
    start_split->y() >> conv_middle2 >> batch_middle2 >> act_middle2;

    auto *bottleneck_result = &bottleneck1->y();
    if (this->num_type == 3)
        bottleneck_result = &((*bottleneck_result) >> bottleneck2 >> bottleneck3);

    cat->forward(*bottleneck_result, act_middle2->y());
    cat->y() >> conv_end >> batch_end >> act_end;

}

LayerCSPDarknet::LayerCSPDarknet(const TensorGpu &x, Workspace &workspace, std::function<void(int)> callback)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;

    try {
        // stem(Focus)
        if (callback != nullptr) callback(0);
        *this += conv_41 = new LayerConvolutional(x, *_workspace, 48, 3, 3, 1, 1, false);
        *this += batch_41 = new LayerBatchNorm(conv_41->y(), 0.001);
        *this += act_41 = new LayerActivationSilu(batch_41->y());
        if (callback != nullptr) callback(3);

        // dark2
        *this += dark2 = new LayerConvolutional(act_41->y(), *_workspace, 96, 3, 3, 2, 1, false);
        *this += dark2_batch = new LayerBatchNorm(dark2->y(), 0.001);
        *this += dark2_act = new LayerActivationSilu(dark2_batch->y());
        if (callback != nullptr) callback(6);

        *this += CSPLayer1 = new CSPLayer(dark2_act->y(), *_workspace, 96, 1, true);
        if (callback != nullptr) callback(7);

        // dark3
        *this += dark3 = new LayerConvolutional(CSPLayer1->result(), *_workspace, 192, 3, 3, 2, 1, false);
        *this += dark3_batch = new LayerBatchNorm(dark3->y(), 0.001);
        *this += dark3_act = new LayerActivationSilu(dark3_batch->y());
        if (callback != nullptr) callback(10);

        *this += CSPLayer2 = new CSPLayer(dark3_act->y(), *_workspace, 192, 3, true);
        if (callback != nullptr) callback(11);
        // dark4
        *this += dark4 = new LayerConvolutional(CSPLayer2->result(), *_workspace, 384, 3, 3, 2, 1, false);
        *this += dark4_batch = new LayerBatchNorm(dark4->y(), 0.001);
        *this += dark4_act = new LayerActivationSilu(dark4_batch->y());
        if (callback != nullptr) callback(14);

        *this += CSPLayer3 = new CSPLayer(dark4_act->y(), *_workspace, 384, 3, true);
        if (callback != nullptr) callback(15);
        // SPPBottleneck
        *this += conv_181 = new LayerConvolutional(CSPLayer3->result(), *_workspace, 768, 3, 3, 2, 1, false);
        *this += batch_181 = new LayerBatchNorm(conv_181->y(), 0.001);
        *this += act_181 = new LayerActivationSilu(batch_181->y());
        if (callback != nullptr) callback(18);

        *this += conv_184 = new LayerConvolutional(act_181->y(), *_workspace, 384, 1, 1, 1, 0, false);
        *this += batch_184 = new LayerBatchNorm(conv_184->y(), 0.001);
        *this += act_184 = new LayerActivationSilu(batch_184->y());
        *this += maxpool_187 = new LayerPooling(act_184->y(), 5, 1, 2, true);   // 2 2 -> 4 4
        *this += maxpool_188 = new LayerPooling(act_184->y(), 9, 1, 4, true);   // 4 4 -> 8 8
        *this += maxpool_189 = new LayerPooling(act_184->y(), 13, 1, 6, true);  // 6 6 -> 12 12
        if (callback != nullptr) callback(24);

        *this += concat_190_1 = new LayerConcat(act_184->y(), maxpool_187->y()); 
        *this += concat_190_2 = new LayerConcat(concat_190_1->y(), maxpool_188->y());
        *this += concat_190_3 = new LayerConcat(concat_190_2->y(), maxpool_189->y());
        if (callback != nullptr) callback(27);
        // dark5
        *this += conv_191 = new LayerConvolutional(concat_190_3->y(), *_workspace, 768, 1, 1, 1, 0, false);
        *this += batch_191 = new LayerBatchNorm(conv_191->y(), 0.001);
        *this += act_191 = new LayerActivationSilu(batch_191->y());
        if (callback != nullptr) callback(30);

        *this += CSPLayer4 = new CSPLayer(act_191->y(), *_workspace, 768, 1, false);  // prob
        if (callback != nullptr) callback(31);
        // FPN
        *this += conv_216 = new LayerConvolutional(CSPLayer4->result(), *_workspace, 384, 1, 1, 1, 0, false);  // lateral_conv0;
        *this += batch_216 = new LayerBatchNorm(conv_216->y(), 0.001);
        *this += act_216 = new LayerActivationSilu(batch_216->y());
        if (callback != nullptr) callback(34);

        *this += up_x0 = new LayerUpSample(act_216->y(), 2);
        *this += concat_221 = new LayerConcat(CSPLayer3->result(), up_x0->y());
        if (callback != nullptr) callback(36);

        *this += CSPLayer5 = new CSPLayer(concat_221->y(), *_workspace, 384, 1, false);
        if (callback != nullptr) callback(37);

        *this += conv_244 = new LayerConvolutional(CSPLayer5->result(), *_workspace, 192, 1, 1, 1, 0, false);  // reduce_conv
        *this += batch_244 = new LayerBatchNorm(conv_244->y(), 0.001);
        *this += act_244 = new LayerActivationSilu(batch_244->y());
        if (callback != nullptr) callback(40);

        *this += up_x1 = new LayerUpSample(act_244->y(), 2);
        *this += concat_249 = new LayerConcat(CSPLayer2->result(), up_x1->y());
        if (callback != nullptr) callback(42);

        *this += CSPLayer6 = new CSPLayer(concat_249->y(), *_workspace, 192, 1, false);
        if (callback != nullptr) callback(43);

        *this += conv_272 = new LayerConvolutional(CSPLayer6->result(), *_workspace, 192, 3, 3, 2, 1, false); // bu_conv2
        *this += batch_272 = new LayerBatchNorm(conv_272->y(), 0.001);
        *this += act_272 = new LayerActivationSilu(batch_272->y());
        *this += concat_275 = new LayerConcat(act_244->y(), act_272->y());
        if (callback != nullptr) callback(47);

        *this += CSPLayer7 = new CSPLayer(concat_275->y(), *_workspace, 384, 1, false);
        if (callback != nullptr) callback(48);

        *this += conv_298 = new LayerConvolutional(CSPLayer7->result(), *_workspace, 384, 3, 3, 2, 1, false); // bu_conv1
        *this += batch_298 = new LayerBatchNorm(conv_298->y(), 0.001);
        *this += act_298 = new LayerActivationSilu(batch_298->y());
        *this += concat_301 = new LayerConcat(act_216->y(), act_298->y());
        if (callback != nullptr) callback(52);

        *this += CSPLayer8 = new CSPLayer(concat_301->y(), *_workspace, 768, 1, false);
        if (callback != nullptr) callback(53);

        *this += cls_convs1_1 = new LayerConvolutional(CSPLayer6->result(), *_workspace, 192, 3, 3, 1, 1, false); // stem1 -> CSPLayer6
        *this += cls_batch1_1 = new LayerBatchNorm(cls_convs1_1->y(), 0.001);
        *this += cls_act1_1 = new LayerActivationSilu(cls_batch1_1->y());
        if (callback != nullptr) callback(56);

        *this += cls_convs1_2 = new LayerConvolutional(cls_act1_1->y(), *_workspace, 192, 3, 3, 1, 1, false);
        *this += cls_batch1_2 = new LayerBatchNorm(cls_convs1_2->y(), 0.001);
        *this += cls_act1_2 = new LayerActivationSilu(cls_batch1_2->y());
        if (callback != nullptr) callback(59);

        *this += cls_convs2_1 = new LayerConvolutional(act_272->y(), *_workspace, 192, 3, 3, 1, 1, false); // stem2 -> conv_272
        *this += cls_batch2_1 = new LayerBatchNorm(cls_convs2_1->y(), 0.001);
        *this += cls_act2_1 = new LayerActivationSilu(cls_batch2_1->y());
        if (callback != nullptr) callback(62);

        *this += cls_convs2_2 = new LayerConvolutional(cls_act2_1->y(), *_workspace, 192, 3, 3, 1, 1, false);
        *this += cls_batch2_2 = new LayerBatchNorm(cls_convs2_2->y(), 0.001);
        *this += cls_act2_2 = new LayerActivationSilu(cls_batch2_2->y());
        if (callback != nullptr) callback(65);

        *this += cls_plus = new LayerPooling(cls_convs2_2->y(), 3, 2, 1, false);
        *this += cls_convs3_1 = new LayerConvolutional(cls_plus->y(), *_workspace, 192, 3, 3, 1, 1, false); // stem3 -> conv_298
        *this += cls_batch3_1 = new LayerBatchNorm(cls_convs3_1->y(), 0.001);
        *this += cls_act3_1 = new LayerActivationSilu(cls_batch3_1->y());
        if (callback != nullptr) callback(69);

        *this += cls_convs3_2 = new LayerConvolutional(cls_act3_1->y(), *_workspace, 192, 3, 3, 1, 1, false);
        *this += cls_batch3_2 = new LayerBatchNorm(cls_convs3_2->y(), 0.001);
        *this += cls_act3_2 = new LayerActivationSilu(cls_batch3_2->y());
        if (callback != nullptr) callback(72);

        *this += reg_convs1_1 = new LayerConvolutional(CSPLayer6->result(), *_workspace, 192, 3, 3, 1, 1, false);
        *this += reg_batch1_1 = new LayerBatchNorm(reg_convs1_1->y(), 0.001);
        *this += reg_act1_1 = new LayerActivationSilu(reg_batch1_1->y());
        if (callback != nullptr) callback(75);

        *this += reg_convs1_2 = new LayerConvolutional(reg_act1_1->y(), *_workspace, 192, 3, 3, 1, 1, false);
        *this += reg_batch1_2 = new LayerBatchNorm(reg_convs1_2->y(), 0.001);
        *this += reg_act1_2 = new LayerActivationSilu(reg_batch1_2->y());
        if (callback != nullptr) callback(78);

        *this += reg_convs2_1 = new LayerConvolutional(act_272->y(), *_workspace, 192, 3, 3, 1, 1, false);
        *this += reg_batch2_1 = new LayerBatchNorm(reg_convs2_1->y(), 0.001);
        *this += reg_act2_1 = new LayerActivationSilu(reg_batch2_1->y());
        if (callback != nullptr) callback(81);

        *this += reg_convs2_2 = new LayerConvolutional(reg_act2_1->y(), *_workspace, 192, 3, 3, 1, 1, false);
        *this += reg_batch2_2 = new LayerBatchNorm(reg_convs2_2->y(), 0.001);
        *this += reg_act2_2 = new LayerActivationSilu(reg_batch2_2->y());
        if (callback != nullptr) callback(84);

       
        *this += reg_plus = new LayerPooling(reg_act2_2->y(), 3, 2, 1, false);
        *this += reg_convs3_1 = new LayerConvolutional(reg_plus->y(), *_workspace, 192, 3, 3, 1, 1, false);
        *this += reg_batch3_1 = new LayerBatchNorm(reg_convs3_1->y(), 0.001);
        *this += reg_act3_1 = new LayerActivationSilu(reg_batch3_1->y());
        if (callback != nullptr) callback(88);

        *this += reg_convs3_2 = new LayerConvolutional(reg_act3_1->y(), *_workspace, 192, 3, 3, 1, 1, false);
        *this += reg_batch3_2 = new LayerBatchNorm(reg_convs3_2->y(), 0.001);
        *this += reg_act3_2 = new LayerActivationSilu(reg_batch3_2->y());
        if (callback != nullptr) callback(91);

        *this += cls_preds1 = new LayerConvolutional(cls_convs1_2->y(), *_workspace, 2, 1, 1, 1, 0, true);  // 80
        *this += cls_preds2 = new LayerConvolutional(cls_convs2_2->y(), *_workspace, 2, 1, 1, 1, 0, true);  // 80 -> car
        *this += cls_preds3 = new LayerConvolutional(cls_convs3_2->y(), *_workspace, 2, 1, 1, 1, 0, true);  // 80
        if (callback != nullptr) callback(94);

        *this += reg_preds1 = new LayerConvolutional(reg_convs1_2->y(), *_workspace, 4, 1, 1, 1, 0, true);
        *this += reg_preds2 = new LayerConvolutional(reg_convs2_2->y(), *_workspace, 4, 1, 1, 1, 0, true);
        *this += reg_preds3 = new LayerConvolutional(reg_convs3_2->y(), *_workspace, 4, 1, 1, 1, 0, true);
        if (callback != nullptr) callback(97);

        *this += obj_preds1 = new LayerConvolutional(reg_convs1_2->y(), *_workspace, 1, 1, 1, 1, 0, true);
        *this += obj_preds2 = new LayerConvolutional(reg_convs2_2->y(), *_workspace, 1, 1, 1, 1, 0, true);
        *this += obj_preds3 = new LayerConvolutional(reg_convs3_2->y(), *_workspace, 1, 1, 1, 1, 0, true);
        if (callback != nullptr) callback(100);

        *this += stem1 = new LayerConvolutional(CSPLayer6->result(), *_workspace, 192, 1, 1, 1, 0, false);
        *this += stem1_batch = new LayerBatchNorm(stem1->y(), 0.001);
        *this += stem1_act = new LayerActivationSilu(stem1_batch->y());
        *this += stem2 = new LayerConvolutional(CSPLayer7->result(), *_workspace, 192, 1, 1, 1, 0, false);
        *this += stem2_batch = new LayerBatchNorm(stem2->y(), 0.001);
        *this += stem2_act = new LayerActivationSilu(stem2_batch->y());
        *this += stem3 = new LayerConvolutional(CSPLayer8->result(), *_workspace, 192, 1, 1, 1, 0, false);
        *this += stem3_batch = new LayerBatchNorm(stem3->y(), 0.001);
        *this += stem3_act = new LayerActivationSilu(stem3_batch->y());
        if (callback != nullptr) callback(109);

        *this += obj_sig1 = new LayerActivationSigmoid(obj_preds1->y());
        *this += obj_sig2 = new LayerActivationSigmoid(obj_preds2->y());
        *this += obj_sig3 = new LayerActivationSigmoid(obj_preds3->y());
        *this += cls_sig1 = new LayerActivationSigmoid(cls_preds1->y());
        *this += cls_sig2 = new LayerActivationSigmoid(cls_preds2->y());
        *this += cls_sig3 = new LayerActivationSigmoid(cls_preds3->y());
        if (callback != nullptr) callback(115);

        *this += out1_cat1 = new LayerConcat(reg_preds1->y(), obj_sig1->y());
        *this += out1_cat2 = new LayerConcat(out1_cat1->y(), cls_sig1->y());
        *this += out2_cat1 = new LayerConcat(reg_preds2->y(), obj_sig2->y());
        *this += out2_cat2 = new LayerConcat(out2_cat1->y(), cls_sig2->y());
        *this += out3_cat1 = new LayerConcat(reg_preds3->y(), obj_sig3->y());
        *this += out3_cat2 = new LayerConcat(out3_cat1->y(), cls_sig3->y());
        if (callback != nullptr) callback(121);

    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");

        tool::TablePrinter printer;
        printer.setHeader({"Layer", "Type", "Output", "Filter", "Option"});
        this->print(printer);
        printer.print();

        throw e;
    }
}
void LayerCSPDarknet::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    // Layer::forward(x);
    
    x >> conv_41 >> batch_41 >> act_41 >> dark2 >> dark2_batch >> dark2_act >> CSPLayer1;
    
    CSPLayer1->result() >> dark3 >> dark3_batch >> dark3_act >> CSPLayer2;
    CSPLayer2->result() >> dark4 >> dark4_batch >> dark4_act >> CSPLayer3;
    CSPLayer3->result() >> conv_181 >> batch_181 >> act_181 >> conv_184 >> batch_184 >> act_184;

    act_184->y() >> maxpool_187;
    act_184->y() >> maxpool_188;
    act_184->y() >> maxpool_189;

    concat_190_1->forward(act_184->y(), maxpool_187->y());
    concat_190_2->forward(concat_190_1->y(), maxpool_188->y());
    concat_190_3->forward(concat_190_2->y(), maxpool_189->y());
    
    concat_190_3->y() >> conv_191 >> batch_191 >> act_191 >> CSPLayer4; 
    
    CSPLayer4->result() >> conv_216 >> batch_216 >> act_216 >> up_x0;
    concat_221->forward(up_x0->y(), CSPLayer3->result());
    concat_221->y() >> CSPLayer5;
    
    CSPLayer5->result() >> conv_244 >> batch_244 >> act_244 >> up_x1;
    concat_249->forward(up_x1->y(), CSPLayer2->result());
    concat_249->y() >> CSPLayer6;
    
    CSPLayer6->result() >> conv_272 >> batch_272 >> act_272;
    concat_275->forward(act_272->y(), act_244->y());
    concat_275->y() >> CSPLayer7;
    
    CSPLayer7->result() >> conv_298 >> batch_298 >> act_298;
    concat_301->forward(act_298->y(), act_216->y());
    concat_301->y() >> CSPLayer8;
    
    CSPLayer6->result() >> stem1 >> stem1_batch >> stem1_act;
    stem1_act->y() >> reg_convs1_1 >> reg_batch1_1 >> reg_act1_1 >> reg_convs1_2 >> reg_batch1_2 >> reg_act1_2;
    reg_act1_2->y() >> reg_preds1;
    reg_act1_2->y() >> obj_preds1 >> obj_sig1;
    stem1_act->y() >> cls_convs1_1 >> cls_batch1_1 >> cls_act1_1 >> cls_convs1_2 >> cls_batch1_2 >> cls_act1_2;
    cls_act1_2->y() >> cls_preds1 >> cls_sig1;

    CSPLayer7->result() >> stem2 >> stem2_batch >> stem2_act;
    stem2_act->y() >> reg_convs2_1 >> reg_batch2_1 >> reg_act2_1 >> reg_convs2_2 >> reg_batch2_2 >> reg_act2_2;
    reg_act2_2->y() >> reg_preds2;
    reg_act2_2->y() >> obj_preds2 >> obj_sig2;
    stem2_act->y() >> cls_convs2_1 >> cls_batch2_1 >> cls_act2_1 >> cls_convs2_2 >> cls_batch2_2 >> cls_act2_2;
    cls_act2_2->y() >> cls_preds2 >> cls_sig2;

    CSPLayer8->result() >> stem3 >> stem3_batch >> stem3_act;
    stem3_act->y() >> reg_convs3_1 >> reg_batch3_1 >> reg_act3_1 >> reg_convs3_2 >> reg_batch3_2 >> reg_act3_2;
    reg_act3_2->y() >> reg_preds3;
    reg_act3_2->y() >> obj_preds3 >> obj_sig3;
    stem3_act->y() >> cls_convs3_1 >> cls_batch3_1 >> cls_act3_1 >> cls_convs3_2 >> cls_batch3_2 >> cls_act3_2;
    cls_act3_2->y() >> cls_preds3 >> cls_sig3;

    out1_cat1->forward(reg_preds1->y(), obj_sig1->y());
    out1_cat2->forward(out1_cat1->y(), cls_sig1->y());
    
    out2_cat1->forward(reg_preds2->y(), obj_sig2->y());
    out2_cat2->forward(out2_cat1->y(), cls_sig2->y());

    out3_cat1->forward(reg_preds3->y(), obj_sig3->y());
    out3_cat2->forward(out3_cat1->y(), cls_sig3->y());
}

} // namespace layer
} // namespace deepnet