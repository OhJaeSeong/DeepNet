/// Copyright (c)2022 HanulSoft(HNS)

#include "deepnet/layer/LayerScrfd.hpp"

#include <iostream>
#include <tuple>
#include <vector>

#include "deepnet/Debug.hpp"

namespace deepnet {
namespace layer {

NormalSeq::NormalSeq(const TensorGpu &x, Workspace &workspace, int input_filter, int kernal, int stride, int pad, int group = 1)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;
    try {
        *this += conv = new LayerConvolutional(x, *_workspace, input_filter, kernal, kernal, stride, pad, false, group);
        *this += batch = new LayerBatchNorm(conv->y(), 1e-5);
        *this += act = new LayerActivationRelu(batch->y());
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void NormalSeq::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    Layer::forward(x);
    x >> conv >> batch >> act;
}

NormalSeqNoact::NormalSeqNoact(const TensorGpu &x, Workspace &workspace, int input_filter, int kernal, int stride, int pad, int group = 1)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;
    try {
        *this += conv = new LayerConvolutional(x, *_workspace, input_filter, kernal, kernal, stride, pad, false, group);
        *this += batch = new LayerBatchNorm(conv->y(), 1e-5);
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void NormalSeqNoact::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    Layer::forward(x);
    x >> conv >> batch;
}

Circle::Circle(const TensorGpu &x, Workspace &workspace,
               int filter1, int filter2)
    : LayerSequential(x), _workspace(&workspace), split(nullptr), merge(nullptr) {
    DEEPNET_TRACER;

    try {
        *this += split = new LayerSplit(x);
        *this += conv1 = new NormalSeq(split->y(), *_workspace, filter1, 1, 1, 0);
        *this += conv2 = new NormalSeq(conv1->y(), *_workspace, filter1, 3, 1, 1);
        *this += conv3 = new NormalSeqNoact(conv2->y(), *_workspace, filter2, 1, 1, 0);
        
        *this += merge = new LayerMerge(conv3->y());
        *this += relu = new layer::LayerActivationRelu(merge->y());
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        // this->print();
        throw e;
    }
}

void Circle::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    Layer::forward(x);
    x >> split >> conv1 >> conv2 >> conv3;
    merge->forward(conv3->y(), split->y());
    merge->y() >> relu;
}


ExpCircle::ExpCircle(const TensorGpu &x, Workspace &workspace,
                     int filter1, int filter2, int str)
    : LayerSequential(x), _workspace(&workspace), split(nullptr), merge(nullptr) {
    DEEPNET_TRACER;

    try {
        *this += split = new LayerSplit(x);
        *this += conv1 = new NormalSeq(x, *_workspace, filter1, 1, 1, 0);
        *this += conv2 = new NormalSeq(conv1->y(), *_workspace, filter1, 3, str, 1);
        *this += conv3 = new NormalSeqNoact(conv2->y(), *_workspace, filter2, 1, 1, 0);    
        *this += avgPool = new LayerPooling(x, str, str, str, str, 0, 0, false);
        *this += conv4 = new NormalSeqNoact(avgPool->y(), *_workspace, filter2, 1, 1, 0);

        *this += merge = new LayerMerge(conv4->y());
        *this += relu = new LayerActivationRelu(merge->y());
    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");
        throw e;
    }
}

void ExpCircle::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    // DEEPNET_ASSERT(&x != nullptr);
    Layer::forward(x);

    x >> split >> conv1;
    conv1->y() >> conv2;
    conv2->y() >> conv3;

    split->y() >> avgPool;
    avgPool->y() >> conv4;
    merge->forward(conv3->y(), conv4->y());
    merge->y() >> relu;
}

LayerScrfd::LayerScrfd(const TensorGpu &x, Workspace &workspace, std::function<void(int)> callback)
    : LayerSequential(x), _workspace(&workspace) {
    DEEPNET_TRACER;

    try {
        if (callback != nullptr) callback(0);
        *this += conv_0 = new NormalSeq(x, *_workspace, 28, 3, 2, 1);
        *this += conv_2 = new NormalSeq(conv_0->y(), *_workspace, 28, 3, 1, 1);

        if (callback != nullptr) callback(2);
        *this += conv_4 = new NormalSeq(conv_2->y(), *_workspace, 56, 3, 1, 1);
        if (callback != nullptr) callback(3);
        *this += maxpool_6 = new LayerPooling(conv_4->y(), 2, 2, 2, 2, 0, 0, true);
        if (callback != nullptr) callback(4);

        *this += ExpCircle1 = new ExpCircle(maxpool_6->y(), *_workspace, 56, 224, 1);
        *this += Circle1 = new Circle(ExpCircle1->y(), *_workspace, 56, 224);
        if (callback != nullptr) callback(6);
        *this += Circle2 = new Circle(Circle1->y(), *_workspace, 56, 224);
        *this += Circle3 = new Circle(Circle2->y(), *_workspace, 56, 224);
        if (callback != nullptr) callback(8);
        *this += Circle4 = new Circle(Circle3->y(), *_workspace, 56, 224);
        *this += Circle5 = new Circle(Circle4->y(), *_workspace, 56, 224);
        if (callback != nullptr) callback(10);
        *this += Circle6 = new Circle(Circle5->y(), *_workspace, 56, 224);
        *this += Circle7 = new Circle(Circle6->y(), *_workspace, 56, 224);
        if (callback != nullptr) callback(12);
        *this += Circle8 = new Circle(Circle7->y(), *_workspace, 56, 224);
        *this += Circle9 = new Circle(Circle8->y(), *_workspace, 56, 224);
        if (callback != nullptr) callback(14);
        *this += Circle10 = new Circle(Circle9->y(), *_workspace, 56, 224);
        *this += Circle11 = new Circle(Circle10->y(), *_workspace, 56, 224);
        if (callback != nullptr) callback(16);
        *this += Circle12 = new Circle(Circle11->y(), *_workspace, 56, 224);
        *this += Circle13 = new Circle(Circle12->y(), *_workspace, 56, 224);
        if (callback != nullptr) callback(18);
        *this += Circle14 = new Circle(Circle13->y(), *_workspace, 56, 224);
        *this += Circle15 = new Circle(Circle14->y(), *_workspace, 56, 224);
        *this += Circle16 = new Circle(Circle15->y(), *_workspace, 56, 224);
        if (callback != nullptr) callback(20);

        *this += ExpCircle2 = new ExpCircle(Circle16->y(), *_workspace, 56, 224, 2);
        *this += Circle17 = new Circle(ExpCircle2->y(), *_workspace, 56, 224);
        if (callback != nullptr) callback(22);
        *this += Circle18 = new Circle(Circle17->y(), *_workspace, 56, 224);
        *this += Circle19 = new Circle(Circle18->y(), *_workspace, 56, 224);
        if (callback != nullptr) callback(24);
        *this += Circle20 = new Circle(Circle19->y(), *_workspace, 56, 224);
        *this += Circle21 = new Circle(Circle20->y(), *_workspace, 56, 224);
        if (callback != nullptr) callback(26);
        *this += Circle22 = new Circle(Circle21->y(), *_workspace, 56, 224);
        *this += Circle23 = new Circle(Circle22->y(), *_workspace, 56, 224);
        if (callback != nullptr) callback(28);
        *this += Circle24 = new Circle(Circle23->y(), *_workspace, 56, 224);
        *this += Circle25 = new Circle(Circle24->y(), *_workspace, 56, 224);
        if (callback != nullptr) callback(30);
        *this += Circle26 = new Circle(Circle25->y(), *_workspace, 56, 224);
        *this += Circle27 = new Circle(Circle26->y(), *_workspace, 56, 224);
        if (callback != nullptr) callback(32);
        *this += Circle28 = new Circle(Circle27->y(), *_workspace, 56, 224);
        *this += Circle29 = new Circle(Circle28->y(), *_workspace, 56, 224);
        if (callback != nullptr) callback(34);
        *this += Circle30 = new Circle(Circle29->y(), *_workspace, 56, 224);
        *this += Circle31 = new Circle(Circle30->y(), *_workspace, 56, 224);  // shortcut
        if (callback != nullptr) callback(36);

        *this += ExpCircle3 = new ExpCircle(Circle31->y(), *_workspace, 144, 576, 2);
        *this += Circle32 = new Circle(ExpCircle3->y(), *_workspace, 144, 576);  // shortcut
        if (callback != nullptr) callback(38);
        *this += ExpCircle4 = new ExpCircle(Circle32->y(), *_workspace, 184, 736, 2);
        *this += Circle33 = new Circle(ExpCircle4->y(), *_workspace, 184, 736);
        if (callback != nullptr) callback(40);
        *this += Circle34 = new Circle(Circle33->y(), *_workspace, 184, 736);
        *this += Circle35 = new Circle(Circle34->y(), *_workspace, 184, 736);
        if (callback != nullptr) callback(42);
        *this += Circle36 = new Circle(Circle35->y(), *_workspace, 184, 736);
        *this += Circle37 = new Circle(Circle36->y(), *_workspace, 184, 736);
        if (callback != nullptr) callback(44);
        *this += Circle38 = new Circle(Circle37->y(), *_workspace, 184, 736);
        *this += Circle39 = new Circle(Circle38->y(), *_workspace, 184, 736);
        if (callback != nullptr) callback(46);

    } catch (std::exception &e) {
        DEEPNET_LOG("Error while constructing:");

        tool::TablePrinter printer;
        printer.setHeader({"Layer", "Type", "Output", "Filter", "Option"});
        this->print(printer);
        printer.print();

        throw e;
    }
}
void LayerScrfd::forward(const TensorGpu &x) {
    DEEPNET_TRACER;
    // Layer::forward(x);
    x >> conv_0 >> conv_2 >> conv_4 >> maxpool_6;
    maxpool_6->y() >> ExpCircle1 >> Circle1 >> Circle2 >> Circle3 >> Circle4 >> Circle5 >> Circle6 >> Circle7 >> Circle8;
    Circle8->y() >> Circle9 >> Circle10 >> Circle11 >> Circle12 >> Circle13 >> Circle14 >> Circle15 >> Circle16;

    Circle16->y() >> ExpCircle2 >> Circle17 >> Circle18 >> Circle19 >> Circle20 >> Circle21 >> Circle22 >> Circle23;
    Circle23->y() >> Circle24 >> Circle25 >> Circle26 >> Circle27 >> Circle28 >> Circle29 >> Circle30 >> Circle31;

    Circle31->y() >> ExpCircle3 >> Circle32;

    Circle32->y() >> ExpCircle4 >> Circle33 >> Circle34 >> Circle35 >> Circle36 >> Circle37 >> Circle38 >> Circle39;
}

} // namespace layer
} // namespace deepnet