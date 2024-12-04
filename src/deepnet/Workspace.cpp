/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/Workspace.hpp"

namespace deepnet {

Workspace::~Workspace() {
    DEEPNET_TRACER;

    if (_workspace) {
        UNSAFE_CUDA(cudaFree(_workspace));
        _workspace = nullptr;
    }
}

void Workspace::enlarge(size_t required_workspace_size) {
    DEEPNET_TRACER;

    // 이미 충분한 공간이 있으면,
    if (_size >= required_workspace_size)
        return;

    // 기존 작업 공간을 삭제한다.
    if (_workspace)
        cudaFree(_workspace);

    // 새 작업 공간을 할당한다.
    SAFE_CUDA(cudaMalloc(&_workspace, required_workspace_size));

    // 작업 공간의 크기를 갱신한다.
    _size = required_workspace_size;
}

} // namespace deepnet
