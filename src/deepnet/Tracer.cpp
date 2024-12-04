/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/Tracer.hpp"
#include <iostream>

namespace deepnet {

std::vector<std::string> Tracer::callstack;

void Tracer::printStack(void) {
    if (Tracer::callstack.size() > 0)
        std::cout << "\x1B[33mCall stack:\x1B[37m" << std::endl;

    for (auto str : Tracer::callstack)
        std::cout << "\x1B[95m" << str << "\x1B[37m" << std::endl;
}

} // namespace deepnet
