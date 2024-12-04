/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include <vector>
#include "Debug.hpp"

/// 0부터 주어진 값(포함하지 않음)까지 일련의 배열을 반환한다.
template <class T>
std::vector<T> range(int end)
{
    DEEPNET_ASSERT(0 <= end);

    std::vector<T> result;
    result.reserve(end);

    for (auto i = 0; i < end; i++)
    {
        result.push_back((T)i);
    }

    return result;
}

/// 시작값부터 끝 값(포함하지 않음)까지 일련의 배열을 반환한다.
template <class T>
std::vector<T> range(int start, int end)
{
    DEEPNET_ASSERT(start <= end);

    std::vector<T> result;
    result.reserve(end - start);

    for (auto i = start; i < end; i++)
    {
        result.push_back((T)i);
    }

    return result;
}

/// 벡터의 값을 터미널에 출력한다.
template <class T>
void print(const std::vector<T> &r)
{
    auto count = r.size();

    std::cout << "[";

    int index = 0;
    for (auto &item : r) {
        std::cout << item;

        if (++index < count)
            std::cout << ", ";
    }

    std::cout << "]" << std::endl;
}
