/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include "deepnet/BaseCuda.hpp"

namespace deepnet {

/// 레이어의 연산에 필요한 작업 공간을 공유하기 위한 클래스.
class Workspace : public BaseCuda {
    /// 작업 공간의 포인터.
    void *_workspace;

    /// 작업 공간으로 할당된 크기.
    size_t _size;

  public:
    /// 생성자.
    Workspace() : _workspace(nullptr), _size(0) {}

    /// 소멸자.
    virtual ~Workspace();

    /// 필요하면 작업 공간을 확장한다.
    void enlarge(size_t required_workspace_size);

    /// 작업 공간의 포인터를 얻는다.
    void *data(void) { return _workspace; }

    /// 작업 공간의 포인터를 얻는다.
    size_t size(void) const { return _size; }
};

} // namespace deepnet
