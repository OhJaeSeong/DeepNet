/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/Weight.hpp"
#include "deepnet/Workspace.hpp"
#include "deepnet/layer/LayerSequential.hpp"
#include <vector>

namespace deepnet {

/// 네트워크 클래스.
class Network : public layer::LayerSequential {
    /// 작업 공간.
    Workspace _workspace;

  protected:
    /// Darknet 형식의 가중치 파일의 헤더를 스킵한다.
    void skipDarknetHeader(FILE *file);

    /// Darknet 형식의 가중치 파일의 헤더를 생성한다.
    void writeDarknetHeader(FILE *file) const;

  public:
    /// 생성자.
    Network(const TensorGpu &x) : LayerSequential(x) {}

    /// 가중치 정보를 읽는다.
    /// @return 성공하면 true.
    bool readWeight(const char *file_path, Weight::Format format);

    /// 가중치 정보를 저장한다.
    /// @return 성공하면 true.
    bool writeWeight(const char *file_path, Weight::Format format) const;

    /// 작업공간 객체를 반환한다.
    inline Workspace &workspace(void) { return _workspace; }

    /// 네트워크의 연산 정보를 출력한다.
    void print(void) const;

    /// 네트워크의 가중치 정보를 출력한다.
    void printWeight(void) const;

    /// 레이어 출력 정보를 콘솔에 출력한다.
    void debug(void) const;
};

} // namespace deepnet
