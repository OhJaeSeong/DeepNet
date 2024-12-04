/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/Image.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/Tensor.hpp"

#if FEATURE_USE_OPENCV == 1
namespace deepnet {

bool Image::read(const char *file_path,               //
                 cv::Mat &original, cv::Mat &resized, //
                 TensorCpu &tensor, int batch) {
    // 멀티쓰레드에서 호출하는 함수에는 TRACER를 사용할 수 없다.
    // DEEPNET_TRACER;

    std::cout << file_path << std::endl;

#ifdef _WINDOWS
    FILE *file = nullptr;
    auto error = fopen_s(&file, file_path, "rb");
    DEEPNET_ASSERT(!error && file != 0);
#else
    FILE *file = fopen(file_path, "rb");
    DEEPNET_ASSERT(file != 0);
#endif // _WINDOWS

    fclose(file);

    // 영상을 읽는다. 7 ms.
    original = cv::imread(file_path);

    if (original.empty()) {
        DEEPNET_LOG("Cannot read file " << file_path);
        return false;
    }

    // 크기를 정규화한다. 21 ms.
    auto height = tensor.height();
    auto width = tensor.width();

    if (original.rows == height && original.cols == width)
        resized = original;
    else
        cv::resize(original, resized, cv::Size(height, width));

    // 텐서로 변환한다. 3 ms.
    tensor.from(resized, batch);
    tensor *= (1.0f/255.0f);

    return true;
}

bool Image::read(const char *file_path, TensorCpu &tensor, int batch, bool normalize) {
    // 멀티쓰레드에서 호출하는 함수에는 TRACER를 사용할 수 없다.
    // DEEPNET_TRACER;

    // 영상을 읽는다. 7 ms.
    auto cv_image = cv::imread(file_path);

    if (cv_image.empty()) {
        DEEPNET_LOG("Cannot read file " << file_path);
        return false;
    }

    // 크기를 정규화한다. 21 ms.
    auto height = tensor.height();
    auto width = tensor.width();

    if (cv_image.rows != height && cv_image.cols != width)
        cv::resize(cv_image, cv_image, cv::Size(height, width));

    tensor.from(cv_image, batch);

    if (normalize)
        tensor *= (1.0f/255.0f);

    return true;
}

} // namespace deepnet
#endif