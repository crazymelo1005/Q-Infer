// dense 设备执行器的 CUDA 装配（阶段 A dense 接线）：把已验收的整层设备入口填进 DenseExec。
// 只有 -DQINFER_ENABLE_CUDA=ON 时才编这一支；主机构建用不到它（block_layer 的 s.dense 恒空）。
#include "dense/dense_exec_cuda.hpp"

#include "dense/gdn_layer_cuda.hpp"
#include "dense/qsa_layer_cuda.hpp"

namespace qinfer::dense {

DenseExec dense_exec_cuda() {
    DenseExec e;
    e.run_gdn = &run_gdn_layer_cuda;
    e.run_qsa = &run_qsa_layer_cuda;
    e.kv = kv_device_exec();
    return e;
}

}  // namespace qinfer::dense