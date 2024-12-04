/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Debug.hpp"
#include "deepnet/Features.hpp"
#if FEATURE_USE_OPENCV == 1
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <thread>

namespace deepnet {

/// 동영상 파일을 읽는 클래스.
/// 멀티 쓰레드 방식을 채택하여 다음 프레임을 미리 버퍼에 읽는다.
class Video : public cv::VideoCapture {
    /// 영상을 읽는 쓰레드.
    std::thread *_reader;

    /// 버퍼에 동시 접근을 막기 위한 뮤텍스.
    std::mutex _mutex;

    /// 읽은 영상을 저장하는 버퍼.
    std::vector<cv::Mat> _buffer;

    /// 쓰레드 종료 여부.
    bool _stop;

    /// 버퍼의 최대 크기.
    int _buffersize;

    /// 영상을 읽는 쓰레드 함수.
    static void reader_thread(Video *video);

  public:
    /// 생성자.
    Video(int buffersize = 10)
        : _reader(nullptr), //
          _stop(false),     //
          _buffersize(buffersize) {
        DEEPNET_ASSERT(buffersize > 0);
    }

    /// 생성자.
    Video(const char *file_path, int buffersize = 10) : Video(buffersize) {
        open(file_path);
    }

    /// 소멸자.
    ~Video() { release(); }

    /// 영상 파일을 연다. 영상 읽기 쓰레드를 실행한다.
    bool open(const char *file_path);

    /// 영상 파일을 닫는다. 영상 읽기 쓰레드를 종료한다.
    virtual void release(void);

    /// 버퍼에 영상이 있는지 확인한다.
    bool isReady(void) const { return _buffer.size() > 0; }

    /// 영상 파일을 읽는다.
    bool read(cv::Mat &image);
};

} // namespace deepnet
#endif