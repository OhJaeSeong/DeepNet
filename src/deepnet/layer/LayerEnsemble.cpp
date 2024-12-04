/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerEnsemble.hpp"
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

EnsembleDownC::EnsembleDownC(const TensorGpu &x, Workspace &workspace, int filter1, int filter2)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;

    try {
        *this += conv1 = new LayerConvolutional(x, *_workspace, filter1, 1, 1, 1, 0, true);
        *this += act1 = new LayerActivationSilu(conv1->y()); // -> split
        *this += conv2 = new LayerConvolutional(act1->y(), *_workspace, filter2, 3, 3, 2, 1, true);
        *this += act2 = new LayerActivationSilu(conv2->y()); // -> split
        
        *this += maxpool = new LayerPooling(x, 2, 2, 0, true);
        *this += conv3 = new LayerConvolutional(maxpool->y(), *_workspace, filter2, 1, 1, 1, 0, true);
        *this += act3 = new LayerActivationSilu(conv3->y());

        *this += cat2layer = new LayerConcat(act2->y(), act3->y());

    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void EnsembleDownC::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    Layer::forward(x);
    x >> conv1 >> act1 >> conv2 >> act2;
    x >> maxpool >> conv3 >> act3;
    cat2layer->forward(act2->y(), act3->y());
}

EnsembleCircleA::EnsembleCircleA(const TensorGpu &x, Workspace &workspace, int input_filter)
    : LayerSequential(x), _workspace(&workspace) { // input_filter = 64, 128
    DEEPNET_TRACER;

    try {
        *this += conv1 = new LayerConvolutional(x, *_workspace, input_filter, 1, 1, 1, 0, true);
        *this += act1 = new LayerActivationSilu(conv1->y()); // -> split
        *this += conv2 = new LayerConvolutional(x, *_workspace, input_filter, 1, 1, 1, 0, true);
        *this += act2 = new LayerActivationSilu(conv2->y()); // -> split
        
        *this += conv3 = new LayerConvolutional(act2->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act3 = new LayerActivationSilu(conv3->y());
        *this += conv4 = new LayerConvolutional(act3->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act4 = new LayerActivationSilu(conv4->y()); // -> split
        
        *this += conv5 = new LayerConvolutional(act4->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act5 = new LayerActivationSilu(conv5->y());
        *this += conv6 = new LayerConvolutional(act5->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act6 = new LayerActivationSilu(conv6->y()); // -> split
        *this += conv7 = new LayerConvolutional(act6->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act7 = new LayerActivationSilu(conv7->y());
        *this += conv8 = new LayerConvolutional(act7->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act8 = new LayerActivationSilu(conv8->y());
        
        *this += cat1 = new LayerConcat(act8->y(), act6->y());
        *this += cat2 = new LayerConcat(cat1->y(), act6->y());
        *this += cat3 = new LayerConcat(cat2->y(), act4->y());
        *this += cat4 = new LayerConcat(cat3->y(), act2->y());
        
        *this += conv9 = new LayerConvolutional(cat4->y(), *_workspace, int(input_filter * 2.5), 1, 1, 1, 0, true);
        *this += act9 = new LayerActivationSilu(conv9->y());

    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void EnsembleCircleA::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    Layer::forward(x);
    x >> conv1 >> act1;
    x >> conv2 >> act2;
    act2->y() >> conv3 >> act3 >> conv4 >> act4;
    act4->y() >> conv5 >> act5 >> conv6 >> act6;
    act6->y() >> conv7 >> act7 >> conv8 >> act8;

    cat1->forward(act8->y(), act6->y());
    cat2->forward(cat1->y(), act4->y());
    cat3->forward(cat2->y(), act2->y());
    cat4->forward(cat3->y(), act1->y());

    cat4->y() >> conv9 >> act9;
}

EnsembleCircleB::EnsembleCircleB(const TensorGpu &x, Workspace &workspace, int input_filter)
    : LayerSequential(x), _workspace(&workspace) { // input_filter = 64, 128
    DEEPNET_TRACER;

    try {
        *this += conv1 = new LayerConvolutional(x, *_workspace, input_filter * 2, 1, 1, 1, 0, true);
        *this += act1 = new LayerActivationSilu(conv1->y()); // -> split
        *this += conv2 = new LayerConvolutional(x, *_workspace, input_filter * 2, 1, 1, 1, 0, true);
        *this += act2 = new LayerActivationSilu(conv2->y()); // -> split
        
        *this += conv3 = new LayerConvolutional(act2->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act3 = new LayerActivationSilu(conv3->y()); // -> split
        *this += conv4 = new LayerConvolutional(act3->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act4 = new LayerActivationSilu(conv4->y()); // -> split
        
        *this += conv5 = new LayerConvolutional(act4->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act5 = new LayerActivationSilu(conv5->y()); // -> split
        *this += conv6 = new LayerConvolutional(act5->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act6 = new LayerActivationSilu(conv6->y()); // -> split
        *this += conv7 = new LayerConvolutional(act6->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act7 = new LayerActivationSilu(conv7->y()); // -> split
        *this += conv8 = new LayerConvolutional(act7->y(), *_workspace, input_filter, 3, 3, 1, 1, true);
        *this += act8 = new LayerActivationSilu(conv8->y()); 
        
        *this += cat1 = new LayerConcat(act8->y(), act7->y());
        *this += cat2 = new LayerConcat(cat1->y(), act6->y());
        *this += cat3 = new LayerConcat(cat2->y(), act5->y());
        *this += cat4 = new LayerConcat(cat3->y(), act4->y());
        *this += cat5 = new LayerConcat(cat4->y(), act3->y());
        *this += cat6 = new LayerConcat(cat5->y(), act1->y());
        *this += cat7 = new LayerConcat(cat6->y(), act2->y());
   
        *this += conv9 = new LayerConvolutional(cat7->y(), *_workspace, int(input_filter * 2.5), 1, 1, 1, 0, true);
        *this += act9 = new LayerActivationSilu(conv9->y());

    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void EnsembleCircleB::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    Layer::forward(x);
    x >> conv1 >> act1;
    x >> conv2 >> act2;
    act2->y() >> conv3 >> act3 >> conv4 >> act4 >> conv5 >> act5 >> conv6 >> act6;
    act6->y() >> conv7 >> act7 >> conv8 >> act8;

    cat1->forward(act8->y(), act7->y());
    cat2->forward(cat1->y(), act6->y());
    cat3->forward(cat2->y(), act5->y());
    cat4->forward(cat3->y(), act4->y());
    cat5->forward(cat4->y(), act3->y());
    cat6->forward(cat5->y(), act2->y());
    cat7->forward(cat6->y(), act1->y());

    cat7->y() >> conv9 >> act9;
}

LayerEnsemble::LayerEnsemble(const TensorGpu &x, Workspace &workspace, std::function<void(int)> callback)
    : LayerSequential(x), _workspace(&workspace)
    {
    DEEPNET_TRACER;

    try {
        if (callback != nullptr) callback(0);
        *this += conv_first = new LayerConvolutional(x, *_workspace, 80, 3, 3, 1, 1, true);
        *this += act_first = new LayerActivationSilu(conv_first->y());
        *this += downc1 = new EnsembleDownC(act_first->y(), *_workspace, 80, 80);
        *this += circle1a = new EnsembleCircleA(downc1->result(), *_workspace, 64);
        *this += circle2a = new EnsembleCircleA(downc1->result(), *_workspace, 64);
        *this += add1 = new LayerCalculateEach(circle1a->result(), circle2a->result(), 0);
        
        *this += downc2 = new EnsembleDownC(add1->y(), *_workspace, 160, 160);
        *this += circle3a = new EnsembleCircleA(downc2->result(), *_workspace, 128);
        *this += circle4a = new EnsembleCircleA(downc2->result(), *_workspace, 128);
        *this += add2 = new LayerCalculateEach(circle3a->result(), circle4a->result(), 0); // branch ->
        
        *this += downc3 = new EnsembleDownC(add2->y(), *_workspace, 320, 320);
        *this += circle5a = new EnsembleCircleA(downc3->result(), *_workspace, 256);
        *this += circle6a = new EnsembleCircleA(downc3->result(), *_workspace, 256);
        *this += add3 = new LayerCalculateEach(circle5a->result(), circle6a->result(), 0); // branch ->
        
        *this += downc4 = new EnsembleDownC(add3->y(), *_workspace, 640, 480);
        *this += circle7a = new EnsembleCircleA(downc4->result(), *_workspace, 384);
        *this += circle8a = new EnsembleCircleA(downc4->result(), *_workspace, 384);
        *this += add4 = new LayerCalculateEach(circle7a->result(), circle8a->result(), 0); // branch ->
        
        *this += downc5 = new EnsembleDownC(add4->y(), *_workspace, 960, 640);
        *this += circle9a = new EnsembleCircleA(downc5->result(), *_workspace, 512);
        *this += circle10a = new EnsembleCircleA(downc5->result(), *_workspace, 512);
        *this += add5 = new LayerCalculateEach(circle9a->result(), circle10a->result(), 0);

        // sppcspc
        *this += sp_cv1 = new LayerConvolutional(add5->y(), *_workspace, 640, 1, 1, 1, 0, true);
        *this += sp_act1 = new LayerActivationSilu(sp_cv1->y());
        *this += sp_cv2 = new LayerConvolutional(add5->y(), *_workspace, 640, 1, 1, 1, 0, true);
        *this += sp_act2 = new LayerActivationSilu(sp_cv2->y());
        *this += sp_cv3 = new LayerConvolutional(sp_act1->y(), *_workspace, 640, 3, 3, 1, 1, true);
        *this += sp_act3 = new LayerActivationSilu(sp_cv3->y());
        *this += sp_cv4 = new LayerConvolutional(sp_act3->y(), *_workspace, 640, 1, 1, 1, 0, true);
        *this += sp_act4 = new LayerActivationSilu(sp_cv4->y());

        *this += sp_maxpool1 = new LayerPooling(sp_act4->y(), 5, 1, 2, true);
        *this += sp_maxpool2 = new LayerPooling(sp_act4->y(), 9, 1, 4, true);
        *this += sp_maxpool3 = new LayerPooling(sp_act4->y(), 13, 1, 6, true);
        *this += poolcat1 = new LayerConcat(sp_act4->y(), sp_maxpool1->y());
        *this += poolcat2 = new LayerConcat(poolcat1->y(), sp_maxpool2->y());
        *this += poolcat3 = new LayerConcat(poolcat2->y(), sp_maxpool3->y());

        *this += sp_cv5 = new LayerConvolutional(poolcat3->y(), *_workspace, 640, 1, 1, 1, 0, true);
        *this += sp_act5 = new LayerActivationSilu(sp_cv5->y());
        *this += sp_cv6 = new LayerConvolutional(sp_act5->y(), *_workspace, 640, 3, 3, 1, 1, true);
        *this += sp_act6 = new LayerActivationSilu(sp_cv6->y());
        *this += sp_cat = new LayerConcat(sp_act6->y(), sp_act2->y());
        *this += sp_cv7 = new LayerConvolutional(sp_cat->y(), *_workspace, 640, 1, 1, 1, 0, true);
        *this += sp_act7 = new LayerActivationSilu(sp_cv7->y()); // branch->

        *this += conv410 = new LayerConvolutional(sp_act7->y(), *_workspace, 480, 1, 1, 1, 0, true);
        *this += act410 = new LayerActivationSilu(conv410->y());
        *this += up413 = new LayerUpSample(act410->y(), 2);
        *this += conv414 = new LayerConvolutional(add4->y(), *_workspace, 480, 1, 1, 1, 0, true);
        *this += act414 = new LayerActivationSilu(conv414->y());
        *this += cat417 = new LayerConcat(act414->y(), up413->y());

        *this += circle1b = new EnsembleCircleB(cat417->y(), *_workspace, 192);
        *this += circle2b = new EnsembleCircleB(cat417->y(), *_workspace, 192);
        *this += add6 = new LayerCalculateEach(circle1b->result(), circle2b->result(), 0); // branch ->

        *this += conv475 = new LayerConvolutional(add6->y(), *_workspace, 320, 1, 1, 1, 0, true);
        *this += act475 = new LayerActivationSilu(conv475->y());
        *this += up478 = new LayerUpSample(act475->y(), 2);
        *this += conv479 = new LayerConvolutional(add3->y(), *_workspace, 320, 1, 1, 1, 0, true);
        *this += act479 = new LayerActivationSilu(conv479->y());
        *this += cat482 = new LayerConcat(act479->y(), up478->y());

        *this += circle3b = new EnsembleCircleB(cat482->y(), *_workspace, 128);
        *this += circle4b = new EnsembleCircleB(cat482->y(), *_workspace, 128);
        *this += add7 = new LayerCalculateEach(circle3b->result(), circle4b->result(), 0); // branch ->

        *this += conv540 = new LayerConvolutional(add7->y(), *_workspace, 160, 1, 1, 1, 0, true);
        *this += act540 = new LayerActivationSilu(conv540->y());
        *this += up543 = new LayerUpSample(act540->y(), 2);
        *this += conv544 = new LayerConvolutional(add2->y(), *_workspace, 160, 1, 1, 1, 0, true);
        *this += act544 = new LayerActivationSilu(conv544->y());
        *this += cat547 = new LayerConcat(act544->y(), up543->y());

        *this += circle5b = new EnsembleCircleB(cat547->y(), *_workspace, 64);
        *this += circle6b = new EnsembleCircleB(cat547->y(), *_workspace, 64);
        *this += add8 = new LayerCalculateEach(circle5b->result(), circle6b->result(), 0); // branch

        *this += downc6 = new EnsembleDownC(add8->y(), *_workspace, 160, 160);
        *this += cat615 = new LayerConcat(downc6->result(), add7->y());

        *this += circle7b = new EnsembleCircleB(cat615->y(), *_workspace, 128);
        *this += circle8b = new EnsembleCircleB(cat615->y(), *_workspace, 128);
        *this += add9 = new LayerCalculateEach(circle7b->result(), circle8b->result(), 0); // branch add672

        *this += downc7 = new EnsembleDownC(add9->y(), *_workspace, 320, 240);
        *this += cat683 = new LayerConcat(downc7->result(), add6->y());

        *this += circle9b = new EnsembleCircleB(cat683->y(), *_workspace, 192);
        *this += circle10b = new EnsembleCircleB(cat683->y(), *_workspace, 192);
        *this += add10 = new LayerCalculateEach(circle9b->result(), circle10b->result(), 0); // branch add740

        *this += downc8 = new EnsembleDownC(add10->y(), *_workspace, 480, 320);
        *this += cat751 = new LayerConcat(downc8->result(), sp_act7->y());

        *this += circle11b = new EnsembleCircleB(cat751->y(), *_workspace, 256);
        *this += circle12b = new EnsembleCircleB(cat751->y(), *_workspace, 256);
        *this += add11 = new LayerCalculateEach(circle11b->result(), circle12b->result(), 0); // add808

        *this += conv809 = new LayerConvolutional(add8->y(), *_workspace, 320, 3, 3, 1, 1, true);
        *this += act809 = new LayerActivationSilu(conv809->y());
        *this += conv812 = new LayerConvolutional(add9->y(), *_workspace, 640, 3, 3, 1, 1, true);
        *this += act812 = new LayerActivationSilu(conv812->y());
        *this += conv815 = new LayerConvolutional(add10->y(), *_workspace, 960, 3, 3, 1, 1, true);
        *this += act815 = new LayerActivationSilu(conv815->y());
        *this += conv818 = new LayerConvolutional(add11->y(), *_workspace, 1280, 3, 3, 1, 1, true);
        *this += act818 = new LayerActivationSilu(conv818->y());

        *this += conv_dummpy1 = new LayerConvolutional(circle6b->result(), *_workspace, 320, 3, 3, 1, 1, true);
        *this += act_dummpy1 = new LayerActivationSilu(conv_dummpy1->y());
        *this += conv_dummpy2 = new LayerConvolutional(circle4b->result(), *_workspace, 640, 3, 3, 1, 1, true);
        *this += act_dummpy2 = new LayerActivationSilu(conv_dummpy2->y());
        *this += conv_dummpy3 = new LayerConvolutional(circle2b->result(), *_workspace, 960, 3, 3, 1, 1, true);
        *this += act_dummpy3 = new LayerActivationSilu(conv_dummpy3->y());
        *this += conv_dummpy4 = new LayerConvolutional(sp_act7->y(), *_workspace, 1280, 3, 3, 1, 1, true);
        *this += act_dummpy4 = new LayerActivationSilu(conv_dummpy4->y());

        TensorGpu test_x(1, 48, 1, 1);
        *this += dummpy_anchor = new LayerImplicit(test_x, *_workspace, true);

        *this += conv822 = new LayerConvolutional(act809->y(), *_workspace, 27, 1, 1, 1, 0, true);
        *this += conv894 = new LayerConvolutional(act812->y(), *_workspace, 27, 1, 1, 1, 0, true);
        *this += conv966 = new LayerConvolutional(act815->y(), *_workspace, 27, 1, 1, 1, 0, true);
        *this += conv1038 = new LayerConvolutional(act818->y(), *_workspace, 27, 1, 1, 1, 0, true);

        *this += conv_dummpy5 = new LayerConvolutional(act809->y(), *_workspace, 27, 1, 1, 1, 0, true);
        *this += conv_dummpy6 = new LayerConvolutional(act812->y(), *_workspace, 27, 1, 1, 1, 0, true);
        *this += conv_dummpy7 = new LayerConvolutional(act815->y(), *_workspace, 27, 1, 1, 1, 0, true);
        *this += conv_dummpy8 = new LayerConvolutional(act818->y(), *_workspace, 27, 1, 1, 1, 0, true);

    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");

        tool::TablePrinter printer;
        printer.setHeader({"Layer", "Type", "Output", "Filter", "Option"});
        this->print(printer);
        printer.print();

        throw e;
    }
}

void LayerEnsemble::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    Layer::forward(x);
    x >> conv_first >> act_first >> downc1;

    downc1->result() >> circle1a;
    downc1->result() >> circle2a;
    add1->forward(circle1a->result(), circle2a->result());
    add1->y() >> downc2;

    downc2->result() >> circle3a;
    downc2->result() >> circle4a;
    add2->forward(circle3a->result(), circle4a->result());
    add2->y() >> downc3;

    downc3->result() >> circle5a;
    downc3->result() >> circle6a;
    add3->forward(circle5a->result(), circle6a->result());
    add3->y() >> downc4;

    downc4->result() >> circle7a;
    downc4->result() >> circle8a;
    add4->forward(circle7a->result(), circle8a->result());
    add4->y() >> downc5;

    downc5->result() >> circle9a;
    downc5->result() >> circle10a;
    add5->forward(circle9a->result(), circle10a->result());

    add5->y() >> sp_cv1 >> sp_act1 >> sp_cv3 >> sp_act3 >> sp_cv4 >> sp_act4;
    sp_maxpool1->forward(sp_act4->y());
    sp_maxpool2->forward(sp_act4->y());
    sp_maxpool3->forward(sp_act4->y());
    poolcat1->forward(sp_act4->y(), sp_maxpool1->y());
    poolcat2->forward(poolcat1->y(), sp_maxpool2->y());
    poolcat3->forward(poolcat2->y(), sp_maxpool3->y());
    poolcat3->y() >> sp_cv5 >> sp_act5 >> sp_cv6 >> sp_act6;
    add5->y() >> sp_cv2 >> sp_act2;

    sp_cat->forward(sp_act6->y(), sp_act2->y());
    sp_cat->y() >> sp_cv7 >> sp_act7;
    
    sp_act7->y() >> conv410 >> act410 >> up413;
    add4->y() >> conv414 >> act414;
    cat417->forward(act414->y(), up413->y());
    cat417->y() >> circle1b;
    cat417->y() >> circle2b;
    add6->forward(circle1b->result(), circle2b->result());

    add6->y() >> conv475 >> act475 >> up478;
    add3->y() >> conv479 >> act479;
    cat482->forward(act479->y(), up478->y());
    cat482->y() >> circle3b;
    cat482->y() >> circle4b;
    add7->forward(circle3b->result(), circle4b->result());

    add7->y() >> conv540 >> act540 >> up543;
    add2->y() >> conv544 >> act544;
    cat547->forward(act544->y(), up543->y());
    cat547->y() >> circle5b;
    cat547->y() >> circle6b;
    add8->forward(circle5b->result(), circle6b->result());
    
    add8->y() >> downc6;
    cat615->forward(downc6->result(), add7->y());
    cat615->y() >> circle7b;
    cat615->y() >> circle8b;
    add9->forward(circle7b->result(), circle8b->result());

    add9->y() >> downc7;
    cat683->forward(downc7->result(), add6->y());
    cat683->y() >> circle9b;
    cat683->y() >> circle10b;
    add10->forward(circle9b->result(), circle10b->result());
    
    add10->y() >> downc8;
    cat751->forward(downc8->result(), sp_act7->y());
    cat751->y() >> circle11b;
    cat751->y() >> circle12b;
    add11->forward(circle11b->result(), circle12b->result());

    add8->y() >> conv809 >> act809 >> conv822;
    add9->y() >> conv812 >> act812 >> conv894;
    add10->y() >> conv815 >> act815 >> conv966;
    add11->y() >> conv818 >> act818 >> conv1038;

    circle6b->result() >> conv_dummpy1 >> act_dummpy1;
    circle4b->result() >> conv_dummpy2 >> act_dummpy2;
    circle2b->result() >> conv_dummpy3 >> act_dummpy3;
    sp_act7->y() >> conv_dummpy4 >> act_dummpy4;

}
} // namespace layer
} // namespace deepnet