/// Copyright (c)2021 Electronics and Telecommunications Research Institute(ETRI)

#pragma once

#include <cstdio>
#include <string>
#include <vector>

namespace deepnet {
namespace tool {

/// 테이블 출력 클래스.
class TablePrinter {
    /// 각 열의 폭.
    std::vector<int> _widths;

    /// 테이블의 열.
    std::vector<std::string> _header;

    /// 테이블의 행.
    std::vector<std::vector<std::string>> _rows;

    /// 테이블의 한 행을 출력한다.
    void print(const std::vector<std::string> &row);

    /// 테이블의 구분선을 출력한다.
    void printLine(char sep1 = '-', char sep2 = '|');

  public:
    /// 생성자.
    inline TablePrinter() {}

    /// 테이블 정보를 초기화한다.
    inline void clear(void) {
        _header.clear();
        _rows.clear();
    }

    /// 열을 지정한다.
    inline void setHeader(const std::vector<std::string> &header) {
        _header.clear();
        _header.assign(header.begin(), header.end());
    }

    /// 열을 지정한다.
    inline void setHeader(const std::initializer_list<const char *> &header) {
        _header.clear();
        _header.assign(header.begin(), header.end());
    }

    /// 행을 추가한다.
    inline void addRow(const std::vector<std::string> &row) {
        _rows.push_back(row);
    }

    /// 행을 추가한다.
    inline void addRow(const std::initializer_list<const char *> &row) {
        std::vector<std::string> t;

        for (auto p : row)
            t.push_back(p);

        _rows.push_back(t);
    }

    /// 테이블을 출력한다.
    void print(void);
};

} // namespace tool
} // namespace deepnet
