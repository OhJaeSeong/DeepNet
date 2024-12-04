/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/BaseCudnn.hpp"

namespace deepnet {

/// CUDNN 핸들.
cudnnHandle_t BaseCudnn::cudnn_handle = nullptr;

} // namespace deepnet
