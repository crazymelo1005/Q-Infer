// 测试用的断言。不用 <cassert> 的 assert：Release 构建定义 NDEBUG 会把它整个编译掉，
// 于是测试变成空跑——「全绿」就失去了意义。CHECK 在任何构建类型下都会真的检查。
#pragma once

#include <cstdio>
#include <cstdlib>

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::fprintf(stderr, "CHECK 失败: %s  (%s:%d)\n", #cond, __FILE__, __LINE__); \
            std::abort();                                                            \
        }                                                                            \
    } while (0)
