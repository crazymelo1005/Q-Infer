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
  src/storage/cooccurrence.cpp
  src/storage/expert_cache.cpp
)
target_include_directories(qinfer_storage PUBLIC src)

add_library(qinfer_scheduling
  src/scheduling/budget_arbiter.cpp
  src/scheduling/step_budget.cpp
  src/scheduling/thread_pool.cpp
  src/scheduling/memory_plan.cpp
)
target_include_directories(qinfer_scheduling PUBLIC src)

add_library(qinfer_artifact
  src/artifact/gguf_table.cpp
)
target_include_directories(qinfer_artifact PUBLIC src)

add_library(qinfer_text
  src/text/tokenizer.cpp
)
target_include_directories(qinfer_text PUBLIC src)
target_link_libraries(qinfer_text PUBLIC qinfer_artifact)

add_library(qinfer_serve
  src/serve/sampler.cpp
  src/serve/prompt.cpp
  src/serve/facade.cpp
  src/serve/json.cpp
  src/serve/scheduler.cpp
  src/serve/server.cpp
  src/serve/telemetry.cpp
  src/serve/records.cpp
  src/serve/prompt_cache.cpp
  src/serve/prometheus.cpp
  src/serve/toolcall.cpp
)
target_include_directories(qinfer_serve PUBLIC src)
target_link_libraries(qinfer_serve PUBLIC qinfer_text qinfer_dense qinfer_ple qinfer_storage qinfer_kernels)
# third_party/httplib/httplib.h = cpp-httplib v0.18.6（整份入库，SHA-256 见提交说明）。
# 「裁剪」落实为不定义 CPPHTTPLIB_OPENSSL_SUPPORT 与 CPPHTTPLIB_ZLIB_SUPPORT；线程池由入口按槽位配置。
target_include_directories(qinfer_serve PUBLIC third_party/httplib)

# M1 服务入口（T9）：HTTP 接线层 + 集成脚本（同库下继续加源，可执行单挂 main.cpp）。
# 可执行名沿用产品名 qinfer-server（run.sh 与 main.cpp 的用法说明都按连字符引用）。
add_executable(qinfer-server src/serve/main.cpp)
target_link_libraries(qinfer-server PRIVATE qinfer_serve qinfer_text qinfer_artifact)
add_test(NAME serve_integration
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/serve/run.sh
                 ${CMAKE_BINARY_DIR} 18080)

add_library(qinfer_ple
  src/ple/row_path.cpp
  src/ple/block.cpp
  src/ple/spec.cpp
)
target_include_directories(qinfer_ple PUBLIC src)
target_link_libraries(qinfer_ple PUBLIC qinfer_artifact qinfer_kernels qinfer_storage qinfer_scheduling)

add_library(qinfer_dense
  src/dense/gated_residual.cpp
  src/dense/gdn_preprocess.cpp
  src/dense/gdn_recurrence.cpp
  src/dense/gdn_layer.cpp
  src/dense/qsa_select.cpp
  src/dense/qsa_attend.cpp
  src/dense/rope.cpp
  src/dense/qsa_index.cpp
  src/dense/qsa_indexer.cpp
  src/dense/kv_page_cache.cpp
  src/dense/kv_store.cpp
  src/dense/qsa_qk_prep.cpp
  src/dense/qsa_layer.cpp
  src/dense/layer_map.cpp
  src/dense/model_head.cpp
  src/dense/block_layer.cpp
  src/dense/step_loop.cpp
  src/dense/token_embed.cpp
  src/dense/kv_window.cpp
  src/dense/spec_decode.cpp
  src/dense/mtp_pack.cpp
  src/dense/mtp_entry.cpp
)
target_include_directories(qinfer_dense PUBLIC src)
target_link_libraries(qinfer_dense PUBLIC qinfer_kernels qinfer_artifact qinfer_experts qinfer_ple)

add_library(qinfer_experts
  src/experts/expert_formats.cpp
  src/experts/ffn.cpp
  src/experts/router.cpp
  src/experts/moe_layer.cpp
  src/experts/expert_worker.cpp
  src/experts/compute_group.cpp
  src/experts/expert_source.cpp
  src/experts/layer_dispatch.cpp
  src/experts/step_record.cpp
)
target_include_directories(qinfer_experts PUBLIC src)
target_link_libraries(qinfer_experts PUBLIC qinfer_artifact qinfer_kernels qinfer_storage
                                           qinfer_scheduling)

add_library(qinfer_kernels
  src/kernels/iq4nl.cpp
  src/kernels/iq2xs.cpp
  src/kernels/iq2xs_dot.cpp
  src/kernels/iq2s.cpp
  src/kernels/iq2xxs.cpp
  src/kernels/iq3s.cpp
  src/kernels/iq3xxs.cpp
  src/kernels/iq4xs.cpp
  src/kernels/iq1m.cpp
  src/kernels/q2_0.cpp
  src/kernels/q6k.cpp
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

add_executable(test_step_budget tests/test_step_budget.cpp)
target_link_libraries(test_step_budget PRIVATE qinfer_scheduling)

add_executable(test_memory_plan tests/test_memory_plan.cpp)
target_link_libraries(test_memory_plan PRIVATE qinfer_scheduling)

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE qinfer_text)

add_executable(test_sampler tests/test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE qinfer_serve)

add_executable(test_prompt tests/test_prompt.cpp)
target_link_libraries(test_prompt PRIVATE qinfer_serve)

add_executable(test_facade tests/test_facade.cpp)
target_link_libraries(test_facade PRIVATE qinfer_serve)

# 集成测试的合成 GGUF 发生器：确定性写出 qinfer-serve-fixture.gguf（T9 起集成脚本调用）。
# 只是数据源，不进 ctest；确定性在集成侧用两次 --out + cmp 核对。
add_executable(serve_fixture_gen tests/serve_fixture_gen.cpp)
target_link_libraries(serve_fixture_gen PRIVATE qinfer_dense qinfer_artifact)

add_executable(test_json tests/test_json.cpp)
target_link_libraries(test_json PRIVATE qinfer_serve)

add_executable(test_scheduler tests/test_scheduler.cpp)
target_link_libraries(test_scheduler PRIVATE qinfer_serve)

add_executable(test_telemetry tests/test_telemetry.cpp)
target_link_libraries(test_telemetry PRIVATE qinfer_serve)

# 请求记录环（T14）：/metrics 的 requests/totals 与 /api/requests 的数据源。
add_executable(test_records tests/test_records.cpp)
target_link_libraries(test_records PRIVATE qinfer_serve)

# token 前缀缓存（conversation_cache 数据源）：全串精确匹配的 LRU，/metrics 实读其统计。
add_executable(test_prompt_cache tests/test_prompt_cache.cpp)
target_link_libraries(test_prompt_cache PRIVATE qinfer_serve)

# /metrics 的 Prometheus text 端口（M2 监控）：内容协商 + 三直方图 + vLLM/qinfer 家族映射。
add_executable(test_prometheus tests/test_prometheus.cpp)
target_link_libraries(test_prometheus PRIVATE qinfer_serve)

# 工具调用域（T18）：输出标记解析、OpenAI/Anthropic 工具形状、tools 归一化与生成拆分。
add_executable(test_toolcall tests/test_toolcall.cpp)
target_link_libraries(test_toolcall PRIVATE qinfer_serve)
add_test(NAME toolcall COMMAND test_toolcall)

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

add_executable(test_q6k tests/test_q6k.cpp)
target_link_libraries(test_q6k PRIVATE qinfer_kernels)

add_executable(test_q8_0 tests/test_q8_0.cpp)
target_link_libraries(test_q8_0 PRIVATE qinfer_kernels)

add_executable(test_iq2s tests/test_iq2s.cpp)
target_link_libraries(test_iq2s PRIVATE qinfer_kernels)

add_executable(test_iq2xxs tests/test_iq2xxs.cpp)
target_link_libraries(test_iq2xxs PRIVATE qinfer_kernels)

add_executable(test_iq1m tests/test_iq1m.cpp)
target_link_libraries(test_iq1m PRIVATE qinfer_kernels)

add_executable(test_iq3s tests/test_iq3s.cpp)
target_link_libraries(test_iq3s PRIVATE qinfer_kernels)

add_executable(test_iq4xs tests/test_iq4xs.cpp)
target_link_libraries(test_iq4xs PRIVATE qinfer_kernels)

add_executable(test_iq3xxs tests/test_iq3xxs.cpp)
target_link_libraries(test_iq3xxs PRIVATE qinfer_kernels)

add_library(qinfer_kv
  src/kv/kv_pool.cpp
)
target_include_directories(qinfer_kv PUBLIC src)

add_executable(test_kv_pool tests/test_kv_pool.cpp)
target_link_libraries(test_kv_pool PRIVATE qinfer_kv)

add_executable(test_ngram tests/test_ngram.cpp)
target_link_libraries(test_ngram PRIVATE qinfer_kernels)

add_executable(test_row_cache tests/test_row_cache.cpp)
target_link_libraries(test_row_cache PRIVATE qinfer_storage)

add_executable(test_expert_cache tests/test_expert_cache.cpp)
target_link_libraries(test_expert_cache PRIVATE qinfer_storage)

# 专家缓存的离线重放工具：读参考引擎 --dump-routing 的轨迹，比较三种替换策略的命中率。
# 需要一个真轨迹文件，故不进 ctest；只保证在 CI 里能编译。
add_executable(expert_cache_replay measure/expert_cache_replay.cpp)
target_link_libraries(expert_cache_replay PRIVATE qinfer_storage)

add_executable(gguf_tensors measure/gguf_tensors.cpp)
target_link_libraries(gguf_tensors PRIVATE qinfer_artifact)

# 线程池与主机内核的伸缩微基准（流式带宽 / 分派开销 / 真实 GEMV）：要独占机器才准，故不进 ctest。
add_executable(thread_pool_bench measure/thread_pool_bench.cpp)
target_link_libraries(thread_pool_bench PRIVATE qinfer_experts qinfer_scheduling)

add_executable(test_ple_read_path tests/test_ple_read_path.cpp)
target_link_libraries(test_ple_read_path PRIVATE qinfer_ple)

add_executable(test_ple_block tests/test_ple_block.cpp)
target_link_libraries(test_ple_block PRIVATE qinfer_ple)

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

add_executable(test_bus_share tests/test_bus_share.cpp)
target_link_libraries(test_bus_share PRIVATE qinfer_scheduling)

add_executable(test_expert_worker tests/test_expert_worker.cpp)
target_link_libraries(test_expert_worker PRIVATE qinfer_experts)

add_executable(test_thread_pool tests/test_thread_pool.cpp)
target_link_libraries(test_thread_pool PRIVATE qinfer_experts)

add_executable(test_compute_group tests/test_compute_group.cpp)
target_link_libraries(test_compute_group PRIVATE qinfer_experts)

add_executable(test_layer_dispatch tests/test_layer_dispatch.cpp)
target_link_libraries(test_layer_dispatch PRIVATE qinfer_experts)

add_executable(test_bf16_gemv tests/test_bf16_gemv.cpp)
target_link_libraries(test_bf16_gemv PRIVATE qinfer_experts qinfer_scheduling)

add_executable(test_model_head tests/test_model_head.cpp)
target_link_libraries(test_model_head PRIVATE qinfer_dense qinfer_artifact)

add_executable(test_layer_map tests/test_layer_map.cpp)
target_link_libraries(test_layer_map PRIVATE qinfer_dense qinfer_artifact)

add_executable(test_qsa_layer tests/test_qsa_layer.cpp)
target_link_libraries(test_qsa_layer PRIVATE qinfer_dense qinfer_artifact)

add_executable(test_qsa_qk_prep tests/test_qsa_qk_prep.cpp)
target_link_libraries(test_qsa_qk_prep PRIVATE qinfer_dense)

add_executable(test_kv_page_cache tests/test_kv_page_cache.cpp)
target_link_libraries(test_kv_page_cache PRIVATE qinfer_dense)

add_executable(test_norm tests/test_norm.cpp)
target_link_libraries(test_norm PRIVATE qinfer_dense)

add_executable(test_qsa_indexer tests/test_qsa_indexer.cpp)
target_link_libraries(test_qsa_indexer PRIVATE qinfer_dense)

add_executable(test_qsa_index tests/test_qsa_index.cpp)
target_link_libraries(test_qsa_index PRIVATE qinfer_dense)

add_executable(test_rope tests/test_rope.cpp)
target_link_libraries(test_rope PRIVATE qinfer_dense)

add_executable(test_qsa_attend tests/test_qsa_attend.cpp)
target_link_libraries(test_qsa_attend PRIVATE qinfer_dense)

add_executable(test_qsa_select tests/test_qsa_select.cpp)
target_link_libraries(test_qsa_select PRIVATE qinfer_dense)

add_executable(test_gdn_layer tests/test_gdn_layer.cpp)
target_link_libraries(test_gdn_layer PRIVATE qinfer_dense)

add_executable(test_gdn_recurrence tests/test_gdn_recurrence.cpp)
target_link_libraries(test_gdn_recurrence PRIVATE qinfer_dense)

add_executable(test_gdn_preprocess tests/test_gdn_preprocess.cpp)
target_link_libraries(test_gdn_preprocess PRIVATE qinfer_dense)

add_executable(test_gated_residual tests/test_gated_residual.cpp)
target_link_libraries(test_gated_residual PRIVATE qinfer_dense qinfer_artifact)

add_executable(test_block_layer tests/test_block_layer.cpp)
target_link_libraries(test_block_layer PRIVATE qinfer_dense qinfer_artifact)

add_executable(test_step_loop tests/test_step_loop.cpp)
target_link_libraries(test_step_loop PRIVATE qinfer_dense qinfer_artifact)

add_executable(test_token_embed tests/test_token_embed.cpp)
target_link_libraries(test_token_embed PRIVATE qinfer_dense qinfer_artifact)

add_executable(test_mtp_pack tests/test_mtp_pack.cpp)
target_link_libraries(test_mtp_pack PRIVATE qinfer_dense qinfer_artifact)

add_executable(test_mtp_entry tests/test_mtp_entry.cpp)
target_link_libraries(test_mtp_entry PRIVATE qinfer_dense qinfer_artifact)

add_executable(test_kv_window tests/test_kv_window.cpp)
target_link_libraries(test_kv_window PRIVATE qinfer_dense qinfer_artifact)

add_executable(test_step_record tests/test_step_record.cpp)
target_link_libraries(test_step_record PRIVATE qinfer_experts)

enable_testing()
add_test(NAME page_table COMMAND test_page_table)
add_test(NAME budget_arbiter COMMAND test_budget_arbiter)
add_test(NAME step_budget COMMAND test_step_budget)
add_test(NAME memory_plan COMMAND test_memory_plan)
add_test(NAME tokenizer COMMAND test_tokenizer)
add_test(NAME sampler COMMAND test_sampler)
add_test(NAME prompt COMMAND test_prompt)
add_test(NAME facade COMMAND test_facade)
add_test(NAME json COMMAND test_json)
add_test(NAME scheduler COMMAND test_scheduler)
add_test(NAME telemetry COMMAND test_telemetry)
add_test(NAME records COMMAND test_records)
add_test(NAME prompt_cache COMMAND test_prompt_cache)
add_test(NAME prometheus COMMAND test_prometheus)
add_test(NAME iq4nl COMMAND test_iq4nl)
add_test(NAME iq2xs COMMAND test_iq2xs)
add_test(NAME q8k COMMAND test_q8k)
add_test(NAME iq2xs_dot COMMAND test_iq2xs_dot)
add_test(NAME q2_0 COMMAND test_q2_0)
add_test(NAME q6k COMMAND test_q6k)
add_test(NAME q8_0 COMMAND test_q8_0)
add_test(NAME iq2s COMMAND test_iq2s)
add_test(NAME iq2xxs COMMAND test_iq2xxs)
add_test(NAME iq1m COMMAND test_iq1m)
add_test(NAME iq3s COMMAND test_iq3s)
add_test(NAME iq4xs COMMAND test_iq4xs)
add_test(NAME iq3xxs COMMAND test_iq3xxs)
add_test(NAME ngram COMMAND test_ngram)
add_test(NAME kv_pool COMMAND test_kv_pool)
add_test(NAME row_cache COMMAND test_row_cache)
add_test(NAME expert_cache COMMAND test_expert_cache)
add_test(NAME ple_read_path COMMAND test_ple_read_path)
add_test(NAME ple_block COMMAND test_ple_block)
add_test(NAME gguf_table COMMAND test_gguf_table)
add_test(NAME expert_formats COMMAND test_expert_formats)
add_test(NAME ffn COMMAND test_ffn)
add_test(NAME router COMMAND test_router)
add_test(NAME moe_layer COMMAND test_moe_layer)
add_test(NAME bus_share COMMAND test_bus_share)
add_test(NAME expert_worker COMMAND test_expert_worker)
add_test(NAME thread_pool COMMAND test_thread_pool)
add_test(NAME compute_group COMMAND test_compute_group)
add_test(NAME layer_dispatch COMMAND test_layer_dispatch)
add_test(NAME gated_residual COMMAND test_gated_residual)
add_test(NAME gdn_preprocess COMMAND test_gdn_preprocess)
add_test(NAME gdn_recurrence COMMAND test_gdn_recurrence)
add_test(NAME gdn_layer COMMAND test_gdn_layer)
add_test(NAME qsa_select COMMAND test_qsa_select)
add_test(NAME qsa_attend COMMAND test_qsa_attend)
add_test(NAME rope COMMAND test_rope)
add_test(NAME qsa_index COMMAND test_qsa_index)
add_test(NAME qsa_indexer COMMAND test_qsa_indexer)
add_test(NAME norm COMMAND test_norm)
add_test(NAME kv_page_cache COMMAND test_kv_page_cache)
add_test(NAME qsa_qk_prep COMMAND test_qsa_qk_prep)
add_test(NAME qsa_layer COMMAND test_qsa_layer)
add_test(NAME layer_map COMMAND test_layer_map)
add_test(NAME model_head COMMAND test_model_head)
add_test(NAME block_layer COMMAND test_block_layer)
add_test(NAME step_loop COMMAND test_step_loop)
add_test(NAME token_embed COMMAND test_token_embed)
add_test(NAME mtp_pack COMMAND test_mtp_pack)
add_test(NAME mtp_entry COMMAND test_mtp_entry)
add_test(NAME kv_window COMMAND test_kv_window)
add_test(NAME bf16_gemv COMMAND test_bf16_gemv)
add_test(NAME step_record COMMAND test_step_record)

# ---- CUDA 内核（默认关闭）----
# ADR-008 第 2 条：骨架不依赖 CUDA，CI 在没有 GPU 的 runner 上只编主机侧、照样跑测试。
# 打开时才需要 nvcc，且要单独一份构建目录（例如 -B build/cuda -DQINFER_ENABLE_CUDA=ON）。
option(QINFER_ENABLE_CUDA "构建 CUDA 内核（需要 nvcc）" OFF)
if(QINFER_ENABLE_CUDA)
  enable_language(CUDA)
  # CUDA::cudart 这个导入目标由 FindCUDAToolkit 提供，语言启用本身不会建它。
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # 环境1 与环境2 的卡都是 RTX 5060 Ti（sm_120）；换卡时用 -DCMAKE_CUDA_ARCHITECTURES 覆盖。
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
  endif()

  add_library(qinfer_kernels_cuda
    src/kernels/cuda/q2_0_gemv.cu
    src/kernels/cuda/iq2s_gemv.cu
    src/kernels/cuda/iq_gemv.cu
    src/kernels/cuda/q6k_gemv.cu
    src/kernels/cuda/iq3s_gemv.cu
    src/kernels/cuda/iq4xs_gemv.cu
    src/kernels/cuda/gdn_proj_gemv.cu
  )
  target_include_directories(qinfer_kernels_cuda PUBLIC src)
  target_link_libraries(qinfer_kernels_cuda PUBLIC qinfer_kernels CUDA::cudart)

  add_executable(test_q2_0_cuda tests/test_q2_0_cuda.cpp)
  target_link_libraries(test_q2_0_cuda PRIVATE qinfer_kernels_cuda)
  add_test(NAME q2_0_cuda COMMAND test_q2_0_cuda)

  add_executable(test_iq2s_cuda tests/test_iq2s_cuda.cpp)
  target_link_libraries(test_iq2s_cuda PRIVATE qinfer_kernels_cuda)
  add_test(NAME iq2s_cuda COMMAND test_iq2s_cuda)

  add_executable(test_iq_gemv_cuda tests/test_iq_gemv_cuda.cpp)
  target_link_libraries(test_iq_gemv_cuda PRIVATE qinfer_kernels_cuda)
  add_test(NAME iq_gemv_cuda COMMAND test_iq_gemv_cuda)

  # 阶段 A：GDN/QSA 稠密段设备路径的对照载体（真字节夹具 + 各内核判据）。
  # 链 qinfer_experts（非 CUDA 版）：对照要用主机 quantize_input / run_gemv / format_name。
  add_executable(test_gdn_qsa_cuda tests/test_gdn_qsa_cuda.cpp)
  target_link_libraries(test_gdn_qsa_cuda PRIVATE qinfer_kernels_cuda qinfer_experts)
  add_test(NAME gdn_qsa_cuda COMMAND test_gdn_qsa_cuda)

  # 设备侧的专家前馈（权重常驻）住在 experts 一侧：它要按档分派、并复用主机的激活量化器，
  # 故不能塞进只依赖 kernels 的那个库里。
  add_library(qinfer_experts_cuda
    src/experts/cuda/expert_ffn_cuda.cu
    src/experts/cuda/cuda_worker.cu
  )
  target_include_directories(qinfer_experts_cuda PUBLIC src)
  target_link_libraries(qinfer_experts_cuda PUBLIC qinfer_kernels_cuda qinfer_experts)

  add_executable(test_expert_ffn_cuda tests/test_expert_ffn_cuda.cpp)
  # 这个入口还要跑整模型一步（`run_step_model`），故另连 dense 与 artifact。
  target_link_libraries(test_expert_ffn_cuda PRIVATE qinfer_experts_cuda qinfer_dense qinfer_artifact)
  add_test(NAME expert_ffn_cuda COMMAND test_expert_ffn_cuda)
endif()
