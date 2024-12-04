/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/Debug.hpp"
#include "deepnet/Tensor.hpp"
#if FEATURE_USE_OPENCV == 1
#include <opencv2/opencv.hpp>
#endif

namespace deepnet {

void TensorCpu::resize(size_t new_size) {
    DEEPNET_TRACER;

    if (_size == new_size)
        return;

    // 이전 메모리를 삭제한다.
    if (_data)
        delete[] _data;

    _size = new_size;

    if (_size == 0) {
        _data = nullptr;
        return;
    }

    // 새 메모리를 할당한다.
    _data = new float[_size];
    DEEPNET_ASSERT(_data);
}

TensorCpu::TensorCpu(const TensorGpu &tensor) : TensorCpu(tensor.dimension()) {
    DEEPNET_TRACER;

    this->from(tensor);
}

void TensorCpu::setDimension(int batch, int channel, int height, int width) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(batch >= 0);
    DEEPNET_ASSERT(channel >= 0);
    DEEPNET_ASSERT(height >= 0);
    DEEPNET_ASSERT(width >= 0);

    _batch = batch;
    _channel = channel;
    _height = height;
    _width = width;

    resize((size_t)_batch * _channel * _height * _width);
}

void TensorCpu::from(const TensorGpu &t) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(_size == t.size());

    if (_size > 0 && _data && t.data())
        SAFE_CUDA(cudaMemcpy((void *)_data,         //
                             (void *)t.data(),      //
                             sizeof(float) * _size, //
                             cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

#if FEATURE_USE_OPENCV == 1
void TensorCpu::from(const cv::Mat &mat, int batch_index) {
    // DEEPNET_TRACER;

    DEEPNET_ASSERT(!mat.empty());
    DEEPNET_ASSERT(!isEmpty());

    DEEPNET_ASSERT(batch_index >= 0 && batch_index < _batch);
    DEEPNET_ASSERT(mat.channels() == 3);
    DEEPNET_ASSERT(_channel == 3);
    DEEPNET_ASSERT(mat.rows == _height);
    DEEPNET_ASSERT(mat.cols == _width);

    float *r = _data + batch_index * _channel * _height * _width;
    float *g = r + _height * _width;
    float *b = g + _height * _width;

    // 영상 데이터를 복사한다.
    for (auto y = 0; y < _height; y++) {
        auto *src = mat.ptr<cv::Vec3b>(y);

        for (auto x = 0; x < _width; x++) {
            *b++ = (float)src[x][0];
            *g++ = (float)src[x][1];
            *r++ = (float)src[x][2];
        }
    }
}

void TensorCpu::to(cv::Mat &mat, int batch_index) const {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!mat.empty());
    DEEPNET_ASSERT(!isEmpty());

    DEEPNET_ASSERT(batch_index >= 0 && batch_index < _batch);
    DEEPNET_ASSERT(mat.channels() == 3);
    DEEPNET_ASSERT(_channel == 3);
    DEEPNET_ASSERT(mat.rows == _height);
    DEEPNET_ASSERT(mat.cols == _width);

    float *r = _data + batch_index * _channel * _height * _width;
    float *g = r + _height * _width;
    float *b = g + _height * _width;

    // 영상 데이터를 복사한다.
    for (auto y = 0; y < _height; y++) {
        auto *dest = mat.ptr<cv::Vec3b>(y);

        for (auto x = 0; x < _width; x++) {
            dest[x][0] = (unsigned char)(*b++ * 255.0f);
            dest[x][1] = (unsigned char)(*g++ * 255.0f);
            dest[x][2] = (unsigned char)(*r++ * 255.0f);
        }
    }
}
#endif

void TensorCpu::fill(float value) {
    DEEPNET_TRACER;

    for (auto i = 0l; i < _size; i++)
        _data[i] = value;
}

void TensorCpu::set(std::vector<float> values) {
    DEEPNET_TRACER;

    auto size = values.size();
    DEEPNET_ASSERT(size > 0 && size <= _size);

    DEEPNET_ASSERT(_data);
    memcpy(_data, values.data(), size * sizeof(float));
}

void TensorCpu::set(const float *const values) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(0 <= _size);
    DEEPNET_ASSERT(_data);

    memcpy(_data, values, _size * sizeof(float));
}

void TensorCpu::operator *=(float value) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!isEmpty());

    auto p = _data;

    for (auto i = 0; i < _size; i++, p++)
        *p = *p * value;
}

void TensorCpu::print(const char *prefix, const char *postfix, //
                      const char *bsep, const char *csep,      //
                      const char *hsep, const char *wsep) const {
    DEEPNET_TRACER;

    std::cout << prefix << "[";

    if (_size > 0) {
        DEEPNET_ASSERT(_data);
        float *p = _data;

        for (auto b = 0; b < _batch; b++) {
            for (auto c = 0; c < _channel; c++) {
                for (auto h = 0; h < _height; h++) {
                    for (auto w = 0; w < _width; w++, p++) {
                        std::cout << *p;
                        if (w < _width - 1)
                            std::cout << wsep;
                    }
                    if (h < _height - 1)
                        std::cout << hsep;
                }
                if (c < _channel - 1)
                    std::cout << csep;
            }
            if (b < _batch - 1)
                std::cout << bsep;
        }
    }

    std::cout << "]" << postfix;
}

#if FEATURE_USE_OPENCV == 1
void TensorCpu::show(const char *title) const {
    DEEPNET_TRACER;

    for (auto i = 0; i < _batch; i++) {
        std::string t = std::string(title) + ":" + std::to_string(i);
        show(t.c_str(), i);
    }
}

void TensorCpu::show(const char *title, int batch_index) const {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!isEmpty());

    cv::Mat mat(_height, _width, CV_8UC3);
    to(mat, batch_index);
    cv::imshow(title, mat);
}
#endif

std::tuple<float, float> TensorCpu::getMinMax(void) const {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!this->isEmpty());

    auto p = _data;
    float min = *p;
    float max = *p;

    for (auto i = 0; i < _size; i++, p++) {
        if (min > *p)
            min = *p;

        if (max < *p)
            max = *p;
    }

    return std::make_tuple(min, max);
}

unsigned char TensorCpu::checksum(void) const {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(!this->isEmpty());

    unsigned char *p = (unsigned char *)_data;
    unsigned char sum = 0;
    size_t size = _size * sizeof(float);

    for (auto i = 0; i < size; i++)
        sum += *p++;

    return ~sum;
}

} // namespace deepnet
