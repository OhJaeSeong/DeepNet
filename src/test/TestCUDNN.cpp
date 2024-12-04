/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/BaseCuda.hpp"
#include "deepnet/BaseCudnn.hpp"
#include "deepnet/Tensor.hpp"
#include "deepnet/range.hpp"
#include "deepnet_test/DeepNetTest.hpp"

using namespace deepnet;

DEEPNET_TEST_BEGIN(TestCUDNN, false) {
    DEEPNET_TRACER;

    std::cout << "CUDNN_MAJOR = " << CUDNN_MAJOR << std::endl;
    std::cout << "CUDNN_MINOR = " << CUDNN_MINOR << std::endl;
    std::cout << "CUDNN_PATCHLEVEL = " << CUDNN_PATCHLEVEL << std::endl;
    std::cout << "CUDNN_VERSION = " << CUDNN_VERSION << std::endl;
}
DEEPNET_TEST_END(TestCUDNN)
