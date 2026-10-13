// dense 设备执行器的 CUDA 装配（阶段 A dense 接线）：返回填好 run_gdn/run_qsa 指针的 DenseExec。
// 只有 -DQINFER_ENABLE_CUDA=ON 时才编这一支——主机构建里这个符号不存在，block_layer 的
// s.dense 恒空，跑纯主机路径。
#pragma once

#include "dense/dense_exec.hpp"

namespace qinfer::dense {

// 客户侧（main.cpp 在 #ifdef QINFER_CUDA_BUILD 下调用）用它把 CUDA 整层入口填进 FacadeConfig。
DenseExec dense_exec_cuda();

}  // namespace qinfer::dense