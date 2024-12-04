/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/Debug.hpp"

#ifdef _WINDOWS
#include <ctime>
#else
#include <time.h>
#endif // _WINDOWS

namespace deepnet {

std::string now2string(void) {
    std::time_t time = std::time(nullptr);
    struct tm time_buffer;

#ifdef _WINDOWS
    localtime_s(&time_buffer, &time);
#else
    localtime_r(&time, &time_buffer);
#endif // _WINDOWS

    char str_buffer[100];
    std::strftime(str_buffer, 100, "%Y-%m-%d %T", &time_buffer);

    return str_buffer;
}

} // namespace deepnet
