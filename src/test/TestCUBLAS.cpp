/// Copyright (c)2021 Electronics and Telecommunications Research
/// Institute(ETRI)

#include "deepnet/BaseCuda.hpp"
#include "deepnet/Tensor.hpp"
#include "deepnet/range.hpp"
#include "deepnet_test/DeepNetTest.hpp"

using namespace deepnet;

DEEPNET_TEST_BEGIN(TestCUBLAS, !deepnet_test::autorun) {
    DEEPNET_TRACER;

    float *A, *B, *C;
    const int m = 2, n = 4, k = 3;
    float alpha = 1.0f, beta = 0.0f;

    DEEPNET_ASSERT(sizeof(float) == 4);

    SAFE_CUDA(cudaMalloc(&A, m * k * sizeof(float)));
    SAFE_CUDA(cudaMalloc(&B, k * n * sizeof(float)));
    SAFE_CUDA(cudaMalloc(&C, m * n * sizeof(float)));

    float aa[m][k] = {{1, 2, 3}, {4, 5, 6}};
    float bb[k][n] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    float cc[m][n];

    SAFE_CUDA(cudaMemcpy(A, &aa[0][0], m * k * sizeof(float),
                         cudaMemcpyKind::cudaMemcpyHostToDevice));
    SAFE_CUDA(cudaMemcpy(B, &bb[0][0], k * n * sizeof(float),
                         cudaMemcpyKind::cudaMemcpyHostToDevice));
    SAFE_CUDA(cudaMemcpy(C, &cc[0][0], m * n * sizeof(float),
                         cudaMemcpyKind::cudaMemcpyHostToDevice));

    // See https://www.programmersought.com/article/59354850870/
    // for more information.

    // Before modification: A^tB^t
    // C(m, n) = A(m, k) x B(k, n)
    //
    // SAFE_CUBLAS(cublasSgemm(
    //     deepnet::BaseCuda::cublas_handle,
    //     CUBLAS_OP_N, CUBLAS_OP_N,
    //     m, n, k,
    //     &alpha,
    //     A, m,
    //     B, k,
    //     &beta,
    //     C, m));

    // After modification: AB = (B^tA^t)^t
    SAFE_CUBLAS(cublasSgemm(deepnet::BaseCuda::cublas_handle, CUBLAS_OP_N,
                            CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C,
                            n));

    SAFE_CUDA(cudaMemcpy(&cc[0][0], C, m * n * sizeof(float),
                         cudaMemcpyKind::cudaMemcpyDeviceToHost));

    SAFE_CUDA(cudaFree(A));
    SAFE_CUDA(cudaFree(B));
    SAFE_CUDA(cudaFree(C));

    for (auto i = 0; i < m; i++) {
        for (auto j = 0; j < n; j++)
            std::cout << cc[i][j] << ", ";

        std::cout << std::endl;
    }
}
DEEPNET_TEST_END(TestCUBLAS)

DEEPNET_TEST_BEGIN(TestSaxpy, false) {
    DEEPNET_TRACER;

    TensorCpu a_cpu(2, 3, 1, 1);
    a_cpu.set(range<float>(10, 16));

    TensorCpu b_cpu(2, 3, 1, 1);
    b_cpu.set(range<float>(6));

    a_cpu.print();
    b_cpu.print();

    TensorGpu a(2, 3, 1, 1);
    TensorGpu b(2, 3, 1, 1);

    a.from(a_cpu);
    b.from(b_cpu);

    // a += b.
    float alpha = 1.0f;
    SAFE_CUBLAS(cublasSaxpy(deepnet::BaseCuda::cublas_handle, (int)a.size(),
                            &alpha, b.data(), 1, a.data(), 1));

    a_cpu.from(a);
    b_cpu.from(b);

    a_cpu.print();
    b_cpu.print();
}
DEEPNET_TEST_END(TestSaxpy)
