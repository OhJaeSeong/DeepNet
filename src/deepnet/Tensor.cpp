/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/Tensor.hpp"
#include "deepnet/Debug.hpp"

namespace deepnet {

Dimension::operator std::string() const {
    return std::to_string(_batch) + ", "   //
           + std::to_string(_channel) + ", " //
           + std::to_string(_height) + ", "  //
           + std::to_string(_width);
}

Tensor::Tensor() : Dimension(), _size(0l), _data(nullptr) {}

void Tensor::reshape(int batch, int channel, int height, int width) {
    DEEPNET_ASSERT(_size == batch * channel * height * width);
    return setDimension(batch, channel, height, width);
}

void Tensor::reshape(const Dimension &dimension) {
    return reshape(dimension.batch(), dimension.channel(), //
                   dimension.height(), dimension.width());
}

} // namespace deepnet
