cmake_minimum_required(VERSION 3.28)
project(qinfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(qinfer_storage
  src/storage/page_table.cpp
)
target_include_directories(qinfer_storage PUBLIC src)

add_library(qinfer_scheduling
  src/scheduling/budget_arbiter.cpp
)
target_include_directories(qinfer_scheduling PUBLIC src)

# 第一片骨架不含 CUDA：三层存储与页表是主机侧代码，CI 在没有 GPU 的 runner 上也能编译并跑测试。
# CUDA 内核与图捕获在序 3 之后引入（见 docs/design/adr/ADR-008）。
add_executable(test_page_table tests/test_page_table.cpp)
target_link_libraries(test_page_table PRIVATE qinfer_storage)

add_executable(test_budget_arbiter tests/test_budget_arbiter.cpp)
target_link_libraries(test_budget_arbiter PRIVATE qinfer_scheduling)

enable_testing()
add_test(NAME page_table COMMAND test_page_table)
add_test(NAME budget_arbiter COMMAND test_budget_arbiter)
