/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/Network.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/Weight.hpp"
#include "deepnet/tool/TablePrinter.hpp"
#include <sstream>

namespace deepnet {

void Network::print(void) const {
    DEEPNET_TRACER;
    
    tool::TablePrinter printer;

    printer.setHeader({"Layer", "Type", "Output", "Filter", "Option"});

    if (_layers.size() > 0) {
        auto *input = _layers[0];
        auto index = 0;

        for (auto *l : _layers)
            l->print(printer, 0, ++index);
    }
    
    printer.print();

    std::cout << "Workspace size = " << _workspace.size() << " bytes." << std::endl;
}

void Network::printWeight(void) const {
    DEEPNET_TRACER;
    
    tool::TablePrinter printer;

    printer.setHeader({"Layer", "Type", "Output", "Min~Max"});

    if (_layers.size() > 0) {
        auto *input = _layers[0];
        auto index = 0;

        for (auto *l : _layers)
            l->printWeight(printer, 0, ++index);
    }
    
    printer.print();
}

void Network::debug(void) const {
    DEEPNET_TRACER;

    if (_layers.size() == 0)
        return;

    tool::TablePrinter printer;

    printer.setHeader({"Layer", "Type", "Output", "Min", "Max", "Data[0:3]"});

    auto *input = _layers[0];

    auto index = 0;
    for (auto *l : _layers) {
        l->debug(printer, 0, ++index);
    }

    printer.print();
}

void Network::skipDarknetHeader(FILE *file) {
    // | Type  | Size | Data     |
    // |-------|------|----------|
    // | int32 |    4 | major    |
    // | int32 |    4 | minor    |
    // | int32 |    4 | revision |
    // | int32 |    4 | seen     |
    // | int32 |    4 | 0        |

    int32_t header[5];

#ifdef _WINDOWS
    auto count = fread_s(header,              //
                         sizeof(int32_t) * 5, //
                         sizeof(int32_t), 5, file);
#else
    auto count = fread(header, //
                       sizeof(int32_t), 5, file);
#endif // _WINDOWS

    DEEPNET_ASSERT(count == 5);
}

void Network::writeDarknetHeader(FILE *file) const {
    // | Type  | Size | Data     |
    // |-------|------|----------|
    // | int32 |    4 | major    |
    // | int32 |    4 | minor    |
    // | int32 |    4 | revision |
    // | int32 |    4 | seen     |
    // | int32 |    4 | 0        |

    int32_t header[5];
    header[0] = header[1] = header[2] = header[3] = header[4] = 0;

    auto count = fwrite(header, sizeof(float), 5, file);
    DEEPNET_ASSERT(count == 5);
}

bool Network::readWeight(const char *file_path, Weight::Format format) {
    DEEPNET_TRACER;

#ifdef _WINDOWS
    FILE *file = nullptr;
    auto error = fopen_s(&file, file_path, "rb");
    if (error || !file) {
        DEEPNET_LOG("fopen error = " << error);
        return false;
    }
#else
    FILE *file = fopen(file_path, "rb");
    if (!file) {
        DEEPNET_LOG("fopen error!");
        return false;
    }
#endif // _WINDOWS

    // Darknet 형식이면 첫 20 바이트를 무시한다.
    if (format == Weight::Format::Darknet)
        skipDarknetHeader(file);

    try {
        layer::LayerSequential::readWeight(file, format);
    } catch (const std::exception &e) {
        DEEPNET_LOG("exception = " << e.what());
        fclose(file);
        return false;
    }

    fclose(file);

    return true;
}

bool Network::writeWeight(const char *file_path, Weight::Format format) const {
    DEEPNET_TRACER;

#ifdef _WINDOWS
    FILE *file = nullptr;
    auto error = fopen_s(&file, file_path, "wb+");
    if (error || !file) {
        DEEPNET_LOG("error = " << error);
        return false;
    }
#else
    FILE *file = fopen(file_path, "wb+");
    if (!file) {
        DEEPNET_LOG("fopen error!");
        return false;
    }
#endif // _WINDOWS

    if (format == Weight::Format::Darknet)
        writeDarknetHeader(file);

    try {
        layer::LayerSequential::writeWeight(file, format);
    } catch (const std::exception &e) {
        DEEPNET_LOG("exception = " << e.what());
        fclose(file);
        return false;
    }

    fclose(file);

    return true;
}

} // namespace deepnet
