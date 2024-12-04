/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include <string>
#include <vector>

namespace deepnet {

/// 함수 호출 스택을 저장하기 위한 클래스.
/// 멀티 쓰레드를 지원하지 않으므로 멀티쓰레드에서 호출되는 함수에서는
/// 사용할 수 없다.
class Tracer {
    static std::vector<std::string> callstack;

  public:
    /// 생성자. 호출 스택 문자열을 배열에 저장한다.
    inline Tracer(const char *here) { //
        callstack.push_back(std::string(here));
    }

    /// 생성자. 호출 스택 문자열을 배열에 저장한다.
    inline Tracer(const char *file, const char *line, const char *function) {
        callstack.push_back(std::string(file) + "(" + line + "): " + function +
                            "()");
    }

    /// 생성자. 호출 스택 문자열을 배열에 저장한다.
    inline Tracer(const char *file, int line, const char *function) { //
        callstack.push_back(std::string(file) + "(" + std::to_string(line) +
                            "): " + function + "()");
    }

    /// 소멸자. 배열에서 호출 스택 문자열을 삭제한다.
    inline ~Tracer() {
        if (callstack.size() > 0)
            callstack.pop_back();
    }

    /// 스택을 출력한다.
    static void printStack(void);
};

} // namespace deepnet
