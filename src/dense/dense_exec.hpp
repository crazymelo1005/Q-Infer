// dense 层的设备执行器（阶段 A dense 接线）：一个函数指针结构，step_loop 在「给了执行器」时把
// GDN/QSA 的整层设备入口交进去，run_block_layer 按层类型分派。
//
// 为什么是函数指针而不是直接调 run_gdn_layer_cuda：block_layer.cpp 编在主机构建里（无 QINFER_CUDA_
// BUILD 宏时 device 分支不编译），而 *_cuda 只在 CUDA 段存在。指针让「主机构建零引用、CUDA 构建由
// 调用方（facade）填」成立——与 BlockScratch::offload 的挂法同构（interfaces §8 的提交面）。
#pragma once

#include <string>

namespace qinfer::artifact {
class GgufFile;
}

namespace qinfer::dense {

struct GdnLayer;
struct GdnScratch;
struct QsaLayer;
struct QsaScratch;

// 一整层 dense 计算的设备实现。两个指针任意一个为空 → 该类型走主机。
struct DenseExec {
    bool (*run_gdn)(const artifact::GgufFile&, const GdnLayer&, const float*, float*, GdnScratch&,
                    std::string&) = nullptr;
    bool (*run_qsa)(const artifact::GgufFile&, const QsaLayer&, const float*, float*, QsaScratch&,
                    std::string&) = nullptr;
};

}  // namespace qinfer::dense