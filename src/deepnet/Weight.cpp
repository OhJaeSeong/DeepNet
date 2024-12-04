/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/Weight.hpp"
#include <cstdlib>
#include <math.h>
#include <time.h>

namespace deepnet {

void Weight::initializeWeight(TensorGpu &filter_gpu, InitMethod method) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!filter_gpu.isEmpty());

    srand((unsigned)time(0));

    TensorCpu cpu(filter_gpu.dimension());

    auto *p = cpu.data();
    auto n = cpu.size();
    DEEPNET_ASSERT(n > 0);

    switch (method) {
    case InitMethod::Xavier: {
        // See Kaiming He et al., 2015.
        auto sqrt_n = 1.0f / (float)sqrt(n);
        auto mininum = -sqrt_n;
        auto range = 2 * sqrt_n;

        for (auto i = 0l; i < n; i++, p++)
            *p = range * (float)rand() / (float)RAND_MAX + mininum;

        break;
    }

    case InitMethod::Xavier2: {
        // Xavier의 2배.
        auto sqrt_n = 2.0f / (float)sqrt(n);
        auto mininum = -sqrt_n;
        auto range = 2 * sqrt_n;

        for (auto i = 0l; i < n; i++, p++)
            *p = range * (float)rand() / (float)RAND_MAX + mininum;

        break;
    }

    case InitMethod::Fill01: {
        // 0.1로 채운다.
        for (auto i = 0l; i < n; i++, p++)
            *p = 0.1f;

        break;
    }

    case InitMethod::FillZero: {
        // 0.1로 채운다.
        for (auto i = 0l; i < n; i++, p++)
            *p = 0.0f;

        break;
    }

    default:
        DEEPNET_ASSERT(!"Unknown method");
    }

    // 초기화된 값을 GPU로 보낸다.
    filter_gpu.from(cpu);
}

void Weight::readWeight(FILE *file, TensorGpu &tensor, Format format) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!tensor.isEmpty());
    TensorCpu cpu(tensor.dimension());
    auto size = cpu.size();

#ifdef _WINDOWS
    auto count =
        fread_s(cpu.data(), sizeof(float) * size, sizeof(float), size, file);
#else
    auto count = fread(cpu.data(), sizeof(float), size, file);
#endif // _WINDOWS

    DEEPNET_ASSERT(count == size);
    tensor.from(cpu);
}

void Weight::writeWeight(FILE *file, const TensorGpu &tensor, Format format) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!tensor.isEmpty());
    TensorCpu cpu(tensor);
    auto size = cpu.size();

    auto count = fwrite(cpu.data(), sizeof(float), size, file);
    DEEPNET_ASSERT(count == size);
}

} // namespace deepnet
