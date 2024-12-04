/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/Trainer.hpp"

namespace deepnet {

bool Trainer::isFileExist(const char *file_path) {
    FILE *file;

#ifdef _WINDOWS
    if (fopen_s(&file, file_path, "rb"))
        return false;
#else
    file = fopen(file_path, "rb");
    if (!file)
        return false;
#endif // _WINDOWS

    fclose(file);
    return true;
}

int Trainer::argmax(const float *p, int size) {
    if (size <= 0)
        return -1;

    float max_value = *p;
    int max_index = 0;

    for (auto index = 1; index < size; index++) {
        p++;
        if (max_value < *p) {
            max_index = index;
            max_value = *p;
        }
    }

    return max_index;
}

std::vector<int> Trainer::argmax(const TensorCpu &x) {

    auto batch = x.batch();
    std::vector<int> result;
    result.reserve(batch);

    float *p = x.data();
    auto count = x.height() * x.width() * x.channel();

    for (auto b = 0; b < batch; b++, p += count) {
        auto index = argmax(p, count);
        result.push_back(index);
    }

    return result;
}

} // namespace deepnet
