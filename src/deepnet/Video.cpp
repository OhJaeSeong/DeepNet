/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/Video.hpp"
#include <chrono>

#if FEATURE_USE_OPENCV == 1
namespace deepnet {

void Video::reader_thread(Video *video) {

    while (!video->_stop) {
        cv::Mat image;

        // 버퍼가 꽉 찾으면,
        if (video->_buffer.size() >= video->_buffersize) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // 영상을 읽는다.
        if (!video->cv::VideoCapture::read(image)) {
            video->_stop = true;
            continue;
        }

        // 영상을 버퍼에 저장한다.
        video->_mutex.lock();
        video->_buffer.push_back(image);
        video->_mutex.unlock();
    }
}

bool Video::open(const char *file_path) {
    DEEPNET_TRACER;

    DEEPNET_ASSERT(file_path);

    // 파일이 열려 있으면 기존 쓰레드를 종료한다.
    release();

    if (!cv::VideoCapture::open(file_path))
        return false;

    _stop = false;
    _reader = new std::thread(reader_thread, this);

    return true;
}

void Video::release(void) {
    DEEPNET_TRACER;

    if (isOpened()) {
        _stop = true;

        if (_reader) {
            _reader->join();
            delete _reader;
            _reader = nullptr;
        }

        cv::VideoCapture::release();
        _buffer.clear();
    }
}

bool Video::read(cv::Mat &image) {
    DEEPNET_TRACER;

    if (!isOpened())
        return false;

    if (_buffer.size() == 0) {
        if (_stop)
            release();

        return false;
    }

    // 첫번째 영상을 읽는다.
    image = std::move(_buffer[0]);

    // 첫 번째 영상을 버퍼에서 삭제한다.
    _mutex.lock();
    _buffer.erase(_buffer.begin(), _buffer.begin() + 1);
    _mutex.unlock();

    return true;
}

} // namespace deepnet
#endif