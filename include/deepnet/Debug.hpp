/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Tracer.hpp"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#define MINI_STRINGIFY(x) #x
#define MINI_TOSTRING(x) MINI_STRINGIFY(x)

#define DEEPNET_LOG_HERE                                                          \
    std::cout << "\x1B[33m"                                                    \
              << __FILE__ "(" MINI_TOSTRING(__LINE__) ")\x1B[37m" << std::endl

#define DEEPNET_LOG(msg)                                                          \
    std::cout << "\x1B[33m"                                                    \
              << __FILE__ "(" MINI_TOSTRING(__LINE__) "): " << msg             \
              << "\x1B[37m" << std::endl

#define DEEPNET_LOG_TIME(msg)                                                     \
    std::cout << "\x1B[33m" << __FILE__ "(" MINI_TOSTRING(__LINE__) "): ["     \
              << deepnet::now2string() << "] " << msg << "\x1B[37m"            \
              << std::endl

#ifdef _DEBUG

#define MINI_PASTE1(x, y) x##y
#define MINI_PASTE2(x, y) MINI_PASTE1(x, y)

#define DEEPNET_TRACER                                                            \
    auto MINI_PASTE2(__tracer__, __LINE__) =                                   \
        deepnet::Tracer(__FILE__, __LINE__, __FUNCTION__)

#define DEEPNET_ASSERT(condition)                                                 \
    if (!(condition)) {                                                        \
        DEEPNET_LOG(#condition);                                                  \
        deepnet::Tracer::printStack();                                         \
        throw std::runtime_error(                                              \
            __FILE__ "(" MINI_TOSTRING(__LINE__) "): " #condition);            \
    }

#define DEEPNET_ASSERT_EQ(value, target, error)                                   \
    DEEPNET_ASSERT(deepnet::approximately_equal(value, target, error))

#define DEEPNET_ASSERT_MSG(condition, message)                                    \
    if (!(condition)) {                                                        \
        DEEPNET_LOG(#condition << " : " << message);                              \
        deepnet::Tracer::printStack();                                         \
        throw std::runtime_error(                                              \
            __FILE__ "(" MINI_TOSTRING(__LINE__) "): " #condition);            \
    }

#else // _DEBUG

#define DEEPNET_TRACER

#define DEEPNET_ASSERT(condition)                                                 \
    if (!(condition)) {                                                        \
        DEEPNET_LOG(#condition);                                                  \
        throw std::runtime_error(                                              \
            __FILE__ "(" MINI_TOSTRING(__LINE__) "): " #condition);            \
    }

#define DEEPNET_ASSERT_EQ(value, target, error)                                   \
    DEEPNET_ASSERT(deepnet::approximately_equal(value, target, error))

#define DEEPNET_ASSERT_MSG(condition, message)                                    \
    if (!(condition)) {                                                        \
        DEEPNET_LOG(#condition << " : " << message);                              \
        throw std::runtime_error(                                              \
            __FILE__ "(" MINI_TOSTRING(__LINE__) "): " #condition);            \
    }

#endif // _DEBUG

namespace deepnet {

// 시간을 문자열로 변경한다.
std::string now2string(void);

template <class T> bool approximately_equal(T value, T target, T error) {
    T diff = value - target;
    if (diff < 0)
        diff = -diff;

    return diff < error;
}

} // namespace deepnet
