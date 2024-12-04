/// Copyright (c)2021 Electronics and Telecommunications Research
/// Institute(ETRI)

#include "deepnet/BaseCuda.hpp"
#include "deepnet/BaseCudnn.hpp"
#include "deepnet/Tensor.hpp"
#include "deepnet/range.hpp"
#include "deepnet_test/DeepNetTest.hpp"

using namespace deepnet;

DEEPNET_TEST_BEGIN(TestCUDAVersion, !deepnet_test::autorun) {
    DEEPNET_TRACER;

    std::cout << "CUDART_VERSION = " << CUDART_VERSION << std::endl;
}
DEEPNET_TEST_END(TestCUDAVersion)

void test_cuda_function(float *y, float *x, int size);

DEEPNET_TEST_BEGIN(TestCUDAFunction, !deepnet_test::autorun) {
    DEEPNET_TRACER;

    const int SIZE = 10;
    float input[10], output[10];
    float *device_input, *device_output;

    for (int i = 0; i < 10; i++)
        input[i] = i * 1.0f;

    for (int i = 0; i < 10; i++)
        std::cout << input[i] << ", ";

    std::cout << std::endl;

    cudaMalloc((void **)&device_input, sizeof(float) * SIZE);
    cudaMalloc((void **)&device_output, sizeof(float) * SIZE);

    cudaMemcpy(device_input, input, SIZE * sizeof(float),
               cudaMemcpyHostToDevice);

    test_cuda_function(device_output, device_input, SIZE);

    cudaMemcpy(output, device_output, SIZE * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++)
        std::cout << output[i] << ", ";

    std::cout << std::endl;

    cudaFree(device_input);
    cudaFree(device_output);

    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}
DEEPNET_TEST_END(TestCUDAFunction)
