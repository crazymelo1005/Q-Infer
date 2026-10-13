// 稠密投影的按档设备 GEMV 统一入口（阶段 A：GDN 三投影与 QSA 四投影共用）。
//
// 与主机 `experts::run_gemv` 同构：按 MatrixSpec.format 分派到设备内核——激活先按该档的搭档量化
// （256 值块的档配 Q8_K、32/64 值块的档配 Q8_0，见 src/experts/ffn.cpp 的 quantize_input），再调
// 对应档的设备 GEMV。支持七档：Q2_0（Q8_0 搭档，专家 down）、IQ2_S / IQ2_XXS / IQ1_M（Q8_K 搭档，
// 专家 gate/up）、Q6_K / IQ3_S / IQ4_XS（Q8_K 搭档，稠密投影档）。档无设备内核或几何非法 → 返回
// 假并写 err，不猜。
//
// 为什么单独建这个入口而不是直接用各档的 _cuda 函数：调用方（gdn_layer_cuda / qsa_layer_cuda）
// 只拿到 MatrixSpec + FfnScratch，不想自己维护一张「档 → 内核」的表；一张表差一处就是静默错值，
// 与主机侧 run_gemv 同一个教训（S-36）。
#pragma once

#include <cstdint>
#include <string>

#include <cuda_runtime.h>  // cudaStream_t

#include "experts/ffn.hpp"

namespace qinfer::kernels {

// stream 为 nullptr 时用默认流。sc 是已量化后的激活（act_k / act_32 由 quantize_input 填好）。
bool gdn_proj_gemv(const experts::MatrixSpec& m, const std::uint8_t* w, const experts::FfnScratch& sc,
                   float* out, cudaStream_t stream, std::string& err);

}  // namespace qinfer::kernels