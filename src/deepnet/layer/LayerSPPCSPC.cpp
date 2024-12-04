/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerSPPCSPC.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/layer/LayerActivationSilu.hpp"
#include "deepnet/layer/LayerBatchNorm.hpp"
#include "deepnet/layer/LayerConvolutional.hpp"
#include "deepnet/layer/LayerPooling.hpp"
#include "deepnet/layer/LayerSplit.hpp"
#include "deepnet/layer/LayerUpSample.hpp"
#include <tuple>
#include <math.h>
#include <vector>
#include <algorithm>
#include <iostream>

namespace deepnet {
namespace layer {

CreatedSeq::CreatedSeq(const TensorGpu &x, Workspace &workspace, int input_filter, int kernal, int stride, int pad, int group = 1)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;
    try {
        *this += conv = new LayerConvolutional(x, *_workspace, input_filter, kernal, kernal, stride, pad, false, group);
        *this += batch = new LayerBatchNorm(conv->y(), 0.001);
        *this += act = new LayerActivationSilu(batch->y());

    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void CreatedSeq::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    Layer::forward(x);
    x >> conv;
    conv->y() >> batch;
    batch->y() >> act;
}


CSPALayer::CSPALayer(const TensorGpu &x, Workspace &workspace, int input_filter)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;

    try {
        *this += conv1 = new LayerConvolutional(x, *_workspace, input_filter, 1, 1, 1, 0, false);
        *this += batch1 = new LayerBatchNorm(conv1->y(), 0.001);
        *this += act1 = new LayerActivationSilu(batch1->y());


        *this += conv2 = new LayerConvolutional(x, *_workspace, input_filter, 1, 1, 1, 0, false);
        *this += batch2 = new LayerBatchNorm(conv2->y(), 0.001);
        *this += act2 = new LayerActivationSilu(batch2->y());
        
        *this += bottle1 = new LayerConvolutional(act2->y(), *_workspace, input_filter, 3, 3, 1, 1, false);
        *this += bb1 = new LayerBatchNorm(bottle1->y(), 0.001);
        *this += ba1 = new LayerActivationSilu(bb1->y());

        *this += bottle2 = new LayerConvolutional(ba1->y(), *_workspace, input_filter, 3, 3, 1, 1, false);
        *this += bb2 = new LayerBatchNorm(bottle2->y(), 0.001);
        *this += ba2 = new LayerActivationSilu(bb2->y());

        *this += bottle3 = new LayerConvolutional(ba2->y(), *_workspace, input_filter, 3, 3, 1, 1, false);
        *this += bb3 = new LayerBatchNorm(bottle3->y(), 0.001);
        *this += ba3 = new LayerActivationSilu(bb3->y());

        *this += bottle4 = new LayerConvolutional(ba3->y(), *_workspace, input_filter, 3, 3, 1, 1, false);
        *this += bb4 = new LayerBatchNorm(bottle4->y(), 0.001);
        *this += ba4 = new LayerActivationSilu(bb4->y());


        *this += cat1 = new LayerConcat(act2->y(), act1->y());
        *this += cat2 = new LayerConcat(ba2->y(), cat1->y());
        *this += cat3 = new LayerConcat(ba4->y(), cat2->y());

        *this += conv3 = new LayerConvolutional(cat3->y(), *_workspace, input_filter * 2, 1, 1, 1, 0, false);
        *this += batch3 = new LayerBatchNorm(conv3->y(), 0.001);
        *this += act3 = new LayerActivationSilu(batch3->y());

    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void CSPALayer::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    Layer::forward(x);

    x >> conv1 >> batch1 >> act1;
    x >> conv2 >> batch2 >> act2;
    act2->y() >> bottle1 >> bb1 >> ba1 >> bottle2 >> bb2 >> ba2;
    ba2->y() >> bottle3 >> bb3 >> ba3 >> bottle4 >> bb4 >> ba4;

    cat1->forward(act2->y(), act1->y());
    cat2->forward(ba2->y(), cat1->y());
    cat3->forward(ba4->y(), cat2->y());

    cat3->y() >> conv3 >> batch3 >> act3;
}


CSPBLayer::CSPBLayer(const TensorGpu &x, Workspace &workspace, int input_filter)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;

    try {
        *this += conv1 = new LayerConvolutional(x, *_workspace, input_filter, 1, 1, 1, 0, false);
        *this += batch1 = new LayerBatchNorm(conv1->y(), 0.001);
        *this += act1 = new LayerActivationSilu(batch1->y());


        *this += conv2 = new LayerConvolutional(x, *_workspace, input_filter, 1, 1, 1, 0, false);
        *this += batch2 = new LayerBatchNorm(conv2->y(), 0.001);
        *this += act2 = new LayerActivationSilu(batch2->y());
        
        *this += bottle1 = new LayerConvolutional(act2->y(), *_workspace, int(input_filter/2) , 3, 3, 1, 1, false);
        *this += bb1 = new LayerBatchNorm(bottle1->y(), 0.001);
        *this += ba1 = new LayerActivationSilu(bb1->y());

        *this += bottle2 = new LayerConvolutional(ba1->y(), *_workspace, int(input_filter/2), 3, 3, 1, 1, false);
        *this += bb2 = new LayerBatchNorm(bottle2->y(), 0.001);
        *this += ba2 = new LayerActivationSilu(bb2->y());

        *this += bottle3 = new LayerConvolutional(ba2->y(), *_workspace, int(input_filter/2), 3, 3, 1, 1, false);
        *this += bb3 = new LayerBatchNorm(bottle3->y(), 0.001);
        *this += ba3 = new LayerActivationSilu(bb3->y());

        *this += bottle4 = new LayerConvolutional(ba3->y(), *_workspace, int(input_filter/2), 3, 3, 1, 1, false);
        *this += bb4 = new LayerBatchNorm(bottle4->y(), 0.001);
        *this += ba4 = new LayerActivationSilu(bb4->y());


        *this += cat1 = new LayerConcat(act2->y(), act1->y());
        *this += cat2 = new LayerConcat(ba1->y(), cat1->y());
        *this += cat3 = new LayerConcat(ba2->y(), cat2->y());
        *this += cat4 = new LayerConcat(ba3->y(), cat3->y());
        *this += cat5 = new LayerConcat(ba4->y(), cat4->y());

        *this += conv3 = new LayerConvolutional(cat5->y(), *_workspace, input_filter, 1, 1, 1, 0, false);
        *this += batch3 = new LayerBatchNorm(conv3->y(), 0.001);
        *this += act3 = new LayerActivationSilu(batch3->y());

    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void CSPBLayer::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

    x >> conv1 >> batch1 >> act1; // x1
    x >> conv2 >> batch2 >> act2; // x2
    act2->y() >> bottle1 >> bb1 >> ba1; // x3
    ba1->y() >> bottle2 >> bb2 >> ba2; // x4
    ba2->y() >> bottle3 >> bb3 >> ba3; // x5
    ba3->y() >> bottle4 >> bb4 >> ba4; // x6

    cat1->forward(act2->y(), act1->y());
    cat2->forward(ba1->y(), cat1->y());
    cat3->forward(ba2->y(), cat2->y());
    cat4->forward(ba3->y(), cat3->y());
    cat5->forward(ba4->y(), cat4->y());

    cat5->y() >> conv3 >> batch3 >> act3;
}


LayerCircle::LayerCircle(const TensorGpu &x, Workspace &workspace, int input_filter)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;

    try {
        *this += conv1 = new CreatedSeq(x, *_workspace, input_filter, 3, 1, 1, input_filter);
        *this += conv2 = new CreatedSeq(conv1->y(), *_workspace, input_filter, 1, 1, 0);
        *this += conv3 = new CreatedSeq(conv2->y(), *_workspace, input_filter, 3, 1, 1, input_filter);
        *this += conv4 = new CreatedSeq(conv3->y(), *_workspace, input_filter, 1, 1, 0);
        *this += conv5 = new CreatedSeq(conv4->y(), *_workspace, input_filter, 3, 1, 1, input_filter);
        *this += conv6 = new CreatedSeq(conv5->y(), *_workspace, input_filter, 1, 1, 0);
        *this += conv7 = new CreatedSeq(conv6->y(), *_workspace, input_filter, 3, 1, 1, input_filter);
        *this += conv8 = new CreatedSeq(conv7->y(), *_workspace, input_filter, 1, 1, 0);
        *this += conv9 = new CreatedSeq(conv8->y(), *_workspace, input_filter, 3, 1, 1, input_filter);
        *this += conv10 = new CreatedSeq(conv9->y(), *_workspace, input_filter, 1, 1, 0);
        *this += conv11 = new CreatedSeq(conv10->y(), *_workspace, input_filter, 3, 1, 1, input_filter);

        *this += conv12 = new LayerConvolutional(conv11->y(), *_workspace, 153, 1, 1, 1, 0, true);

    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void LayerCircle::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    Layer::forward(x);
    x >> conv1 >> conv2 >> conv3 >> conv4 >> conv5 >> conv6 >> conv7 >> conv8 >> conv9 >> conv10 >> conv11 >> conv12;
}


LayerSPPCSPC::LayerSPPCSPC(const TensorGpu &x, Workspace &workspace, std::function<void(int)> callback)
    : LayerSequential(x), _workspace(&workspace)
    {
    DEEPNET_TRACER;

    try {
        // stem(Focus)
        if (callback != nullptr) callback(0);
        *this += conv_41 = new CreatedSeq(x, *_workspace, 64, 3, 1, 1);

        // dark2
        *this += conv_45 = new CreatedSeq(conv_41->y(), *_workspace, 128, 3, 2, 1);
        if (callback != nullptr) callback(2);
        *this += CSPALayer1 = new CSPALayer(conv_45->y(), *_workspace, 64);
        if (callback != nullptr) callback(8);
        // dark3
        *this += conv_78 = new CreatedSeq(CSPALayer1->result(), *_workspace, 256, 3, 2, 1);
        if (callback != nullptr) callback(9);
        *this += CSPALayer2 = new CSPALayer(conv_78->y(), *_workspace, 128); // x0
        if (callback != nullptr) callback(15);
        // dark4
        *this += conv_111 = new CreatedSeq(CSPALayer2->result(), *_workspace, 512, 3, 2, 1);
        if (callback != nullptr) callback(16);
        *this += CSPALayer3 = new CSPALayer(conv_111->y(), *_workspace, 256); // x1
        if (callback != nullptr) callback(22);

        *this += conv_144 = new CreatedSeq(CSPALayer3->result(), *_workspace, 768, 3, 2, 1);
        if (callback != nullptr) callback(23);
        *this += CSPALayer4 = new CSPALayer(conv_144->y(), *_workspace, 384); // x2
        if (callback != nullptr) callback(29);

        *this += conv_177 = new CreatedSeq(CSPALayer4->result(), *_workspace, 1024, 3, 2, 1);
        if (callback != nullptr) callback(30);
        *this += CSPALayer5 = new CSPALayer(conv_177->y(), *_workspace, 512); // x3
        if (callback != nullptr) callback(36);

        *this += conv_210 = new CreatedSeq(CSPALayer5->result(), *_workspace, 512, 1, 1, 0);
        *this += conv_234 = new CreatedSeq(CSPALayer5->result(), *_workspace, 512, 1, 1, 0);
        *this += conv_214 = new CreatedSeq(conv_210->y(), *_workspace, 512, 3, 1, 1);
        *this += conv_218 = new CreatedSeq(conv_214->y(), *_workspace, 512, 1, 1, 0);
        if (callback != nullptr) callback(40);

        *this += maxpool_222 = new LayerPooling(conv_218->y(), 5, 1, 2, true); // 2 2 -> 4 4
        *this += maxpool_223 = new LayerPooling(conv_218->y(), 9, 1, 4, true); // 4 4 -> 8 8
        *this += maxpool_224 = new LayerPooling(conv_218->y(), 13, 1, 6, true); // 6 6 -> 12 12

        *this += concat1_1 = new LayerConcat(maxpool_223->y(), maxpool_224->y()); 
        *this += concat1_2 = new LayerConcat(maxpool_222->y(), concat1_1->y());
        *this += concat1_3 = new LayerConcat(conv_218->y(), concat1_2->y());
        if (callback != nullptr) callback(46);

        *this += conv_226 = new CreatedSeq(concat1_3->y(), *_workspace, 512, 1, 1, 0);
        *this += conv_230 = new CreatedSeq(conv_226->y(), *_workspace, 512, 3, 1, 1);
        *this += concat2_1 = new LayerConcat(conv_230->y(), conv_234->y());
        *this += conv_239 = new CreatedSeq(concat2_1->y(), *_workspace, 512, 1, 1, 0);
        if (callback != nullptr) callback(50);


        *this += conv_243 = new CreatedSeq(conv_239->y(), *_workspace, 384, 1, 1, 0); // x4
        *this += up_248 = new LayerUpSample(conv_243->y(), 2);
        *this += conv_249 = new CreatedSeq(CSPALayer4->result(), *_workspace, 384, 1, 1, 0); // x2
        *this += concat3_1 = new LayerConcat(conv_249->y(), up_248->y());
        if (callback != nullptr) callback(54);
        *this += CSPBLayer1 = new CSPBLayer(concat3_1->y(), *_workspace, 384); // x4
        if (callback != nullptr) callback(60);
        
        *this += conv_283 = new CreatedSeq(CSPBLayer1->result(), *_workspace, 256, 1, 1, 0); // x5
        *this += up_288 = new LayerUpSample(conv_283->y(), 2);
        *this += conv_289 = new CreatedSeq(CSPALayer3->result(), *_workspace, 256, 1, 1, 0); // x1
        *this += concat3_2 = new LayerConcat(conv_289->y(), up_288->y());
        if (callback != nullptr) callback(64);
        *this += CSPBLayer2 = new CSPBLayer(concat3_2->y(), *_workspace, 256); // x5
        if (callback != nullptr) callback(70);
        
        *this += conv_323 = new CreatedSeq(CSPBLayer2->result(), *_workspace, 128, 1, 1, 0); // x6
        *this += up_328 = new LayerUpSample(conv_323->y(), 2);
        *this += conv_329 = new CreatedSeq(CSPALayer2->result(), *_workspace, 128, 1, 1, 0); // x0
        *this += concat3_3 = new LayerConcat(conv_329->y(), up_328->y());
        if (callback != nullptr) callback(74);
        *this += CSPBLayer3 = new CSPBLayer(concat3_3->y(), *_workspace, 128); // x6
        if (callback != nullptr) callback(80);

        *this += conv_363 = new CreatedSeq(CSPBLayer3->result(), *_workspace, 256, 3, 2, 1);
        *this += concat4_1 = new LayerConcat(conv_363->y(), CSPBLayer2->result());
        if (callback != nullptr) callback(82);
        *this += CSPBLayer4 = new CSPBLayer(concat4_1->y(), *_workspace, 256);
        if (callback != nullptr) callback(88);
        
        *this += conv_397 = new CreatedSeq(CSPBLayer4->result(), *_workspace, 384, 3, 2, 1);
        *this += concat4_2 = new LayerConcat(conv_397->y(), CSPBLayer1->result());
        if (callback != nullptr) callback(90);
        *this += CSPBLayer5 = new CSPBLayer(concat4_2->y(), *_workspace, 384);
        if (callback != nullptr) callback(96);
        
        *this += conv_431 = new CreatedSeq(CSPBLayer5->result(), *_workspace, 512, 3, 2, 1);
        *this += concat4_3 = new LayerConcat(conv_431->y(), conv_239->y());
        if (callback != nullptr) callback(98);
        *this += CSPBLayer6 = new CSPBLayer(concat4_3->y(), *_workspace, 512);
        if (callback != nullptr) callback(104);

        *this += conv_465 = new CreatedSeq(CSPBLayer3->result(), *_workspace, 256, 3, 1, 1);
        *this += conv_469 = new CreatedSeq(CSPBLayer4->result(), *_workspace, 512, 3, 1, 1);
        *this += conv_473 = new CreatedSeq(CSPBLayer5->result(), *_workspace, 768, 3, 1, 1);
        *this += conv_477 = new CreatedSeq(CSPBLayer6->result(), *_workspace, 1024, 3, 1, 1);
        if (callback != nullptr) callback(108);
        // /model.118.anchors : torch.Size([4, 3, 2]) 618
        // /model.118.anchor_grid : torch.Size([4, 1, 3, 1, 1, 2]) 619
        *this += c0 = new LayerConvolutional(conv_465->y(), *_workspace, 18, 1, 1, 1, 0, true);
        *this += c1 = new LayerConvolutional(conv_469->y(), *_workspace, 18, 1, 1, 1, 0, true);
        *this += c2 = new LayerConvolutional(conv_473->y(), *_workspace, 18, 1, 1, 1, 0, true);
        *this += c3 = new LayerConvolutional(conv_477->y(), *_workspace, 18, 1, 1, 1, 0, true);
        if (callback != nullptr) callback(112);

        *this += a0 = new LayerImplicit(conv_465->y(), *_workspace, true);
        *this += a1 = new LayerImplicit(conv_469->y(), *_workspace, true);
        *this += a2 = new LayerImplicit(conv_473->y(), *_workspace, true);
        *this += a3 = new LayerImplicit(conv_477->y(), *_workspace, true);
        *this += m0 = new LayerImplicit(c0->y(), *_workspace, false);
        *this += m1 = new LayerImplicit(c1->y(), *_workspace, false);
        *this += m2 = new LayerImplicit(c2->y(), *_workspace, false);
        *this += m3 = new LayerImplicit(c3->y(), *_workspace, false);
        if (callback != nullptr) callback(116);

        *this += circle1 = new LayerCircle(conv_465->y(), *_workspace, 256);
        if (callback != nullptr) callback(123);
        *this += circle2 = new LayerCircle(conv_469->y(), *_workspace, 512);
        if (callback != nullptr) callback(130);
        *this += circle3 = new LayerCircle(conv_473->y(), *_workspace, 768);
        if (callback != nullptr) callback(137);
        *this += circle4 = new LayerCircle(conv_477->y(), *_workspace, 1024);
        if (callback != nullptr) callback(144);

        *this += x6_concat = new LayerConcat(m0->y(), circle1->y());
        *this += x7_concat = new LayerConcat(m1->y(), circle2->y());
        *this += x8_concat = new LayerConcat(m2->y(), circle3->y());
        *this += x9_concat = new LayerConcat(m3->y(), circle4->y());
        if (callback != nullptr) callback(148);
        // dark5
        // FPN

        
        std::cout << "model end" << std::endl;
        
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");

        tool::TablePrinter printer;
        printer.setHeader({"Layer", "Type", "Output", "Filter", "Option"});
        this->print(printer);
        printer.print();

        throw e;
    }
}
void LayerSPPCSPC::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    // Layer::forward(x);
    x >> conv_41;
    conv_41->y() >> conv_45;
    conv_45->y() >> CSPALayer1;
    CSPALayer1->result() >> conv_78 >> CSPALayer2; // x0
    CSPALayer2->result() >> conv_111 >> CSPALayer3; // x1
    CSPALayer3->result() >> conv_144 >> CSPALayer4; // x2
    CSPALayer4->result() >> conv_177 >> CSPALayer5; // x3

    CSPALayer5->result() >> conv_234; // x3_c
    CSPALayer5->result() >> conv_210 >> conv_214 >> conv_218;
    conv_218->y() >> maxpool_222;
    conv_218->y() >> maxpool_223;
    conv_218->y() >> maxpool_224;
    concat1_1->forward(maxpool_223->y(), maxpool_224->y());
    concat1_2->forward(maxpool_222->y(), concat1_1->y());
    concat1_3->forward(conv_218->y(), concat1_2->y());
    concat1_3->y() >> conv_226 >> conv_230;
    concat2_1->forward(conv_230->y(), conv_234->y());
    concat2_1->y() >> conv_239; // x3

    conv_239->y() >> conv_243 >> up_248;
    CSPALayer4->result() >> conv_249;
    concat3_1->forward(conv_249->y(), up_248->y());
    concat3_1->y() >> CSPBLayer1; // x4

    CSPBLayer1->result() >> conv_283 >> up_288;
    CSPALayer3->result() >> conv_289;
    concat3_2->forward(conv_289->y(), up_288->y());
    concat3_2->y() >> CSPBLayer2; // x5

    CSPBLayer2->result() >> conv_323  >> up_328;
    CSPALayer2->result() >> conv_329;
    concat3_3->forward(conv_329->y(), up_328->y());
    concat3_3->y() >> CSPBLayer3; // x6

    CSPBLayer3->result() >> conv_363;
    concat4_1->forward(conv_363->y(), CSPBLayer2->result());
    concat4_1->y() >> CSPBLayer4;

    CSPBLayer4->result() >> conv_397;
    concat4_2->forward(conv_397->y(), CSPBLayer1->result());
    concat4_2->y() >> CSPBLayer5;

    CSPBLayer5->result() >> conv_431;
    concat4_3->forward(conv_431->y(), conv_239->y());
    concat4_3->y() >> CSPBLayer6;


    CSPBLayer3->result() >> conv_465;
    CSPBLayer4->result() >> conv_469;
    CSPBLayer5->result() >> conv_473;
    CSPBLayer6->result() >> conv_477;

    conv_465->y() >> circle1;
    conv_465->y() >> a0 >> c0 >> m0;
    x6_concat->forward(m0->y(), circle1->y());

    conv_469->y() >> circle2;
    conv_469->y() >> a1 >> c1 >> m1;
    x7_concat->forward(m1->y(), circle2->y());

    conv_473->y() >> circle3;
    conv_473->y() >> a2 >> c2 >> m2;
    x8_concat->forward(m2->y(), circle3->y());

    conv_477->y() >> circle4;
    conv_477->y() >> a3 >> c3 >> m3;
    x9_concat->forward(m3->y(), circle4->y());

    // x2 = self.conv_249(x2)
    }
} // namespace layer
} // namespace deepnet