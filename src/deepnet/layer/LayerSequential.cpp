/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/layer/LayerSequential.hpp"
#include <sstream>

// #define TIMER

#ifdef TIMER
#include "deepnet/Timer.hpp"
#define LOG_TIME                                                               \
    std::cout << __LINE__ << ": " << timer.elapsed() << std::endl;             \
    timer.start();
#else // TIMER
#define LOG_TIME
#endif // TIMER

namespace deepnet {
namespace layer {

void LayerSequential::train(void) {
    LayerWeighted::train();

    for (auto *l : _layers) {
        l->train();
    }
}

void LayerSequential::eval(void) {
    LayerWeighted::eval();

    for (auto *l : _layers)
        l->eval();
}

void LayerSequential::forward(const TensorGpu &x) {
    DEEPNET_TRACER;

#ifdef TIMER
    Timer timer;
#endif // TIMER

    auto size = _layers.size();
    DEEPNET_ASSERT(size >= 0);
    LOG_TIME;

    const TensorGpu *prev_x = &x;

    LOG_TIME;

    for (int index = 0; index < size; index++) {
        auto *l = _layers[index];

// #ifdef DEBUG
//         std::cout << "Forwarding " << l->type() << "..." << std::endl;
// #endif // DEBUG

        try {
            l->forward(*prev_x);
            LOG_TIME;

        } catch (std::exception &e) {
            DEEPNET_LOG("layer=" << index + 1 << ", x=("      //
                              << prev_x->batch() << ", "   //
                              << prev_x->channel() << ", " //
                              << prev_x->height() << ", "  //
                              << prev_x->width() << ")");

            throw e;
        }

        prev_x = &l->y();
    }
}

void LayerSequential::backward(const TensorGpu &dy) {
    DEEPNET_TRACER;

    // Layer::backward(dy);

    auto size = _layers.size();
    DEEPNET_ASSERT(size >= 0);

    const TensorGpu *next_dy = &dy;
    const TensorGpu *residual_dy = nullptr;

    for (int index = (int)(size - 1); index >= 0; index--) {
        auto *l = _layers[index];

        try {
            l->backward(*next_dy);
        } catch (std::exception &e) {
            DEEPNET_LOG("layer=" << index + 1 << ", y=("        //
                              << l->y().batch() << ", "      //
                              << l->y().channel() << ", "    //
                              << l->y().height() << ", "     //
                              << l->y().width() << "), dy=(" //
                              << next_dy->batch() << ", "    //
                              << next_dy->channel() << ", "  //
                              << next_dy->height() << ", "   //
                              << next_dy->width() << ")");

            throw e;
        }

        next_dy = &l->dx();
    }
}

void LayerSequential::randomizeWeight(Weight::InitMethod method) {
    DEEPNET_TRACER;

    for (auto *l : _layers) {
        auto *p = dynamic_cast<LayerWeighted *>(l);
        if (p)
            p->randomizeWeight(method);
    }
}

void LayerSequential::readWeight(FILE *file, Weight::Format format) {
    DEEPNET_TRACER;

    for (auto *l : _layers) {
        auto *p = dynamic_cast<LayerWeighted *>(l);
        if (p)
            p->readWeight(file, format);
    }
}

void LayerSequential::writeWeight(FILE *file, Weight::Format format) const {
    DEEPNET_TRACER;

    for (auto *l : _layers) {
        auto *p = dynamic_cast<LayerWeighted *>(l);
        if (p)
            p->writeWeight(file, format);
    }
}

const TensorGpu &LayerSequential::y(void) const {
    auto size = _layers.size();
    DEEPNET_ASSERT(size > 0);

    return _layers[size - 1]->y();
}

const TensorGpu &LayerSequential::dx(void) const {
    auto size = _layers.size();
    DEEPNET_ASSERT(size > 0);

    return _layers[0]->dx();
}

void LayerSequential::print(tool::TablePrinter &printer, //
                            int depth, int index) const {
    DEEPNET_TRACER;

    printer.addRow({std::string(depth, '-') + std::to_string(index), //
                    std::string(type()),                             //
                    "", "", ""});

    index = 0;

    for (auto *l : _layers)
        l->print(printer, depth + 1, ++index);
}

void LayerSequential::printWeight(tool::TablePrinter &printer, //
                            int depth, int index) const {
    DEEPNET_TRACER;

    printer.addRow({std::string(depth, '-') + std::to_string(index), //
                    std::string(type()),                             //
                    "", ""});

    index = 0;

    for (auto *l : _layers)
        l->printWeight(printer, depth + 1, ++index);
}

void LayerSequential::debug(tool::TablePrinter &printer, int depth, int index) {
    DEEPNET_TRACER;

    printer.addRow({std::string(depth, '-') + std::to_string(index), //
                    std::string(type()),                             //
                    "", "", "", ""});

    index = 0;

    for (auto *l : _layers)
        l->debug(printer, depth + 1, ++index);
}

void LayerSequential::update(float learning_rate, Weight::UpdateMethod method) {
    for (auto *l : _layers) {
        auto *p = dynamic_cast<LayerWeighted *>(l);
        if (p)
            p->update(learning_rate, method);
    }
}

unsigned char LayerSequential::checksum(void) const {
    unsigned char sum = 0;

    for (auto *l : _layers) {
        auto *p = dynamic_cast<LayerWeighted *>(l);
        if (p)
            sum += ~p->checksum();
    }

    return ~sum;
}

} // namespace layer
} // namespace deepnet
