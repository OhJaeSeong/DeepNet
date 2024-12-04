/// Copyright (c)2021 Electronics and Telecommunications Research
/// Institute(ETRI)

#include "deepnet/Debug.hpp"
#include "deepnet_test/DeepNetTest.hpp"

DEEPNET_TEST_BEGIN(TestColoredText, !deepnet_test::autorun) {
    DEEPNET_TRACER;

    std::cout << "\x1B[30m30 Black\x1B[37m" << std::endl;
    std::cout << "\x1B[31m31 Red\x1B[37m" << std::endl;
    std::cout << "\x1B[32m32 Green\x1B[37m" << std::endl;
    std::cout << "\x1B[33m33 Yellow\x1B[37m" << std::endl;
    std::cout << "\x1B[34m34 Blue\x1B[37m" << std::endl;
    std::cout << "\x1B[35m35 Magenta\x1B[37m" << std::endl;
    std::cout << "\x1B[36m36 Cyan\x1B[37m" << std::endl;
    std::cout << "\x1B[37m37 White\x1B[37m" << std::endl;

    std::cout << "\x1B[90m90 Bright Black\x1B[37m" << std::endl;
    std::cout << "\x1B[91m91 Bright Red\x1B[37m" << std::endl;
    std::cout << "\x1B[92m92 Bright Green\x1B[37m" << std::endl;
    std::cout << "\x1B[93m93 Bright Yellow\x1B[37m" << std::endl;
    std::cout << "\x1B[94m94 Bright Blue\x1B[37m" << std::endl;
    std::cout << "\x1B[95m95 Bright Magenta\x1B[37m" << std::endl;
    std::cout << "\x1B[96m96 Bright Cyan\x1B[37m" << std::endl;
    std::cout << "\x1B[97m97 Bright White\x1B[37m" << std::endl;
}
DEEPNET_TEST_END(TestColoredText)
