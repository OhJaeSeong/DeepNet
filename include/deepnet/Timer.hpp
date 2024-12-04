/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include <chrono>
#include <thread>

namespace deepnet {
/// Timer class.
class Timer {
    std::chrono::time_point<std::chrono::system_clock> _start_time;

  public:
    Timer() { start(); }

    /// 타이머를 시작한다.
    inline void start(void) { _start_time = std::chrono::system_clock::now(); }

    /// 일정 시간동안 멈춘다. 타이머의 동작과는 무관한다.
    inline static void sleep(long milliseconds) {
        std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
    }

    /// 타이머 시작 시간부터 현재까지의 경과한 시간(밀리초)을 얻는다.
    inline long elapsed(bool reset = false) {
        auto stop_time = std::chrono::system_clock::now();
        auto elapsed_time =
            (long)std::chrono::duration_cast<std::chrono::milliseconds>(
                stop_time - _start_time)
                .count();
        
		if (reset)
			_start_time = stop_time;

        return elapsed_time;
    }
};
} // namespace deepnet
