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
  src/storage/row_cache.cpp
)
target_include_directories(qinfer_storage PUBLIC src)

add_library(qinfer_scheduling
  src/scheduling/budget_arbiter.cpp
)
target_include_directories(qinfer_scheduling PUBLIC src)

add_library(qinfer_artifact
  src/artifact/gguf_table.cpp
)
target_include_directories(qinfer_artifact PUBLIC src)

add_library(qinfer_experts
  src/experts/expert_formats.cpp
  src/experts/ffn.cpp
  src/experts/router.cpp
  src/experts/moe_layer.cpp
)
target_include_directories(qinfer_experts PUBLIC src)
target_link_libraries(qinfer_experts PUBLIC qinfer_artifact qinfer_kernels)

add_library(qinfer_kernels
  src/kernels/iq4nl.cpp
  src/kernels/iq2xs.cpp
  src/kernels/iq2xs_dot.cpp
  src/kernels/iq2s.cpp
  src/kernels/iq2xxs.cpp
  src/kernels/q2_0.cpp
  src/kernels/q8_0.cpp
  src/kernels/q8k.cpp
  src/kernels/ngram.cpp
  src/kernels/ple_gather.cpp
)
target_include_directories(qinfer_kernels PUBLIC src)

# 第一片骨架不含 CUDA：三层存储与页表是主机侧代码，CI 在没有 GPU 的 runner 上也能编译并跑测试。
# CUDA 内核与图捕获在序 3 之后引入（见 docs/design/adr/ADR-008）。
add_executable(test_page_table tests/test_page_table.cpp)
target_link_libraries(test_page_table PRIVATE qinfer_storage)

add_executable(test_budget_arbiter tests/test_budget_arbiter.cpp)
target_link_libraries(test_budget_arbiter PRIVATE qinfer_scheduling)

add_executable(test_iq4nl tests/test_iq4nl.cpp)
target_link_libraries(test_iq4nl PRIVATE qinfer_kernels)

add_executable(test_iq2xs tests/test_iq2xs.cpp)
target_link_libraries(test_iq2xs PRIVATE qinfer_kernels)

add_executable(test_q8k tests/test_q8k.cpp)
target_link_libraries(test_q8k PRIVATE qinfer_kernels)

add_executable(test_iq2xs_dot tests/test_iq2xs_dot.cpp)
target_link_libraries(test_iq2xs_dot PRIVATE qinfer_kernels)

add_executable(test_q2_0 tests/test_q2_0.cpp)
target_link_libraries(test_q2_0 PRIVATE qinfer_kernels)

add_executable(test_iq2s tests/test_iq2s.cpp)
target_link_libraries(test_iq2s PRIVATE qinfer_kernels)

add_executable(test_iq2xxs tests/test_iq2xxs.cpp)
target_link_libraries(test_iq2xxs PRIVATE qinfer_kernels)

add_executable(test_ngram tests/test_ngram.cpp)
target_link_libraries(test_ngram PRIVATE qinfer_kernels)

add_executable(test_row_cache tests/test_row_cache.cpp)
target_link_libraries(test_row_cache PRIVATE qinfer_storage)

add_executable(test_ple_read_path tests/test_ple_read_path.cpp)
target_link_libraries(test_ple_read_path PRIVATE qinfer_storage qinfer_scheduling qinfer_kernels)

add_executable(test_gguf_table tests/test_gguf_table.cpp)
target_link_libraries(test_gguf_table PRIVATE qinfer_artifact qinfer_kernels)

add_executable(test_expert_formats tests/test_expert_formats.cpp)
target_link_libraries(test_expert_formats PRIVATE qinfer_experts)

add_executable(test_ffn tests/test_ffn.cpp)
target_link_libraries(test_ffn PRIVATE qinfer_experts)

add_executable(test_router tests/test_router.cpp)
target_link_libraries(test_router PRIVATE qinfer_experts)

add_executable(test_moe_layer tests/test_moe_layer.cpp)
target_link_libraries(test_moe_layer PRIVATE qinfer_experts)

enable_testing()
add_test(NAME page_table COMMAND test_page_table)
add_test(NAME budget_arbiter COMMAND test_budget_arbiter)
add_test(NAME iq4nl COMMAND test_iq4nl)
add_test(NAME iq2xs COMMAND test_iq2xs)
add_test(NAME q8k COMMAND test_q8k)
add_test(NAME iq2xs_dot COMMAND test_iq2xs_dot)
add_test(NAME q2_0 COMMAND test_q2_0)
add_test(NAME iq2s COMMAND test_iq2s)
add_test(NAME iq2xxs COMMAND test_iq2xxs)
add_test(NAME ngram COMMAND test_ngram)
add_test(NAME row_cache COMMAND test_row_cache)
add_test(NAME ple_read_path COMMAND test_ple_read_path)
add_test(NAME gguf_table COMMAND test_gguf_table)
add_test(NAME expert_formats COMMAND test_expert_formats)
add_test(NAME ffn COMMAND test_ffn)
add_test(NAME router COMMAND test_router)
add_test(NAME moe_layer COMMAND test_moe_layer)
