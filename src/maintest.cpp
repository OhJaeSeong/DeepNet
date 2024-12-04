/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/BaseCuda.hpp"
#include "deepnet/BaseCudnn.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet_test/DeepNetTest.hpp"

DEEPNET_TEST_MAIN()
#pragma warning(disable:4819)
/// main function.

int main(int argc, char *argv[]) {
    DEEPNET_TRACER;

    try {
        deepnet::BaseCuda::create();
        deepnet::BaseCudnn::create();
    } catch (std::exception &e) {
        DEEPNET_LOG("Error: " << e.what());
        return -1;
    }

    DEEPNET_LOG_TIME("Start.");
    bool success;

    if (argc <= 1)
        success = DEEPNET_TEST_RUN();
    else
        success = DEEPNET_TEST_RUN(argv[1]);

    DEEPNET_LOG_TIME("End.");

    try {
        deepnet::BaseCuda::destroy();
        deepnet::BaseCudnn::destroy();
    } catch (std::exception &e) {
        DEEPNET_LOG("Error: " << e.what());
        return -1;
    }

    return !success;
}
