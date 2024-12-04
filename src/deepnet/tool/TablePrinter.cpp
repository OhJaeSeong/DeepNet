/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#include "deepnet/tool/TablePrinter.hpp"
#include "deepnet/Debug.hpp"
#include "deepnet/Image.hpp"
#include "deepnet/Tensor.hpp"

namespace deepnet {
namespace tool {

void TablePrinter::print(const std::vector<std::string> &row) {
    DEEPNET_TRACER;

    std::cout << "|";
    int index = 0;

    for (auto col : row) {
        std::cout << " " << col;

        auto space = _widths[index] - col.size();
        for (auto i = 0; i < space; i++)
            std::cout << " ";

        std::cout << " |";
        index++;
    }

    std::cout << std::endl;
}

void TablePrinter::printLine(char sep1, char sep2) {
    std::cout << sep2;
    for (auto w : _widths) {
        for (auto i = 0; i < w + 2; i++)
            std::cout << sep1;
        std::cout << sep2;
    }

    std::cout << std::endl;
}

void TablePrinter::print(void) {
    DEEPNET_TRACER;

    _widths.clear();

    // 헤더의 크기에 맞게 폭을 조정한다.
    for (auto col : _header) {
        _widths.push_back((int)col.size());
    }

    // 각 행의 열 폭에 맞게 조정한다.
    for (auto row : _rows) {
        auto index = 0;
        for (auto col : row) {
            auto size = col.size();
            if (_widths[index] < size)
                _widths[index] = (int)size;
            index++;
        }
    }

    printLine('=', '=');
    print(_header);
    printLine();

    for (auto row : _rows)
        print(row);

    printLine('=', '=');
}

} // namespace tool
} // namespace deepnet
