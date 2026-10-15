# Native Engine 乱码修复与多 GPU 协同设计

状态: 设计中  
作者: Q-Infer Team

---

## 1. 问题陈述

### 1.1 乱码问题

NativeBackend 和 Inhouse backend 在真模型（IQ2_XS）上输出乱码：

```
"itches Vis在先astlecheckerpealenYE typical三CLUDrarysdkBABatia statabbit_ct而喻 stack"
```

对比:
- NativeBackend (GPU): 乱码
- Inhouse backend (GPU): 乱码
- Strata 引擎 (同一模型): 输出连贯（根据 runbook 记录）

结论: 问题不在 NativeBackend vs Engine 的区别，而在更底层——量化格式解析、token embedding、或采样器。

### 1.2 GPU 协同问题

当前状态:
| 设备 | 状态 | 利用率 | 职责 |
|------|------|--------|------|
| GPU 0 (RTX 5060 Ti) | 是 工作中 | ~27% | Dense 层 (GDN+QSA) |
| GPU 1 (RTX 5060 Ti) | 否 未使用 | 0% | 闲置 |
| iGPU (Intel Arc) | 否 未接入 | - | P3b 已取消 (ADR-019) |
| CPU (Intel 285K) | 是 工作中 | - | MoE 专家 + 嵌入 + LM head |

目标: GPU 0 + GPU 1 + CPU 全协同，最大化推理吞吐。

---

## 2. 乱码根因调查

### 2.1 可能原因

1. 量化格式解析错误
   - IQ2_XS 的量化表（codebook）读取方式不同
   - PLE（Piecewise Linear Encoding）表解析错误
   - 分片读取逻辑不一致

2. Token Embedding 读取错误
   - Token embedding 表在 GGUF 中的位置/格式
   - 量化 token embedding vs 全精度
   - 词表大小不匹配（248320）

3. LM Head 读取错误
   - LM head 权重读取方式
   - 是否与 token embedding 共享（tied embeddings）

4. Chat Template 渲染差异
   - 模板渲染的 prompt 格式不同
   - 特殊 token 处理
   - BOS/EOS 处理

5. 采样器 Bug
   - Temperature=0 的 greedy 采样逻辑
   - Top-p/top-k 实现
   - Logits 归一化

### 2.2 调查计划

#### Phase 1: 隔离问题层

```bash
# Test 1: 纯 CPU 路径（不用 GPU）
./build/qinfer-server --backend native ...  # 无 --gpu

# Test 2: 对比 token embedding
# 写测试：读取 token 1 的 embedding，对比 Strata 的输出

# Test 3: 对比 chat template 渲染
# 写测试：渲染 "Say hello"，对比 prompt 文本

# Test 4: 对比 logits
# 写测试：输入相同 token，对比 layer 48 输出的 logits 分布
```

#### Phase 2: 逐层对拍

1. Token Embedding 层
   - 输入: token_id = 1 (BOS)
   - 输出: embedding vector (2560 dim)
   - 对比: Strata 的 embedding 输出

2. Dense 层 (GDN/QSA)
   - 输入: embedding
   - 输出: hidden states after 48 layers
   - 对比: GPU vs CPU 路径是否一致

3. MoE 层
   - 输入: hidden states
   - 输出: expert output
   - 对比: 路由逻辑 + 专家计算

4. LM Head 层
   - 输入: final hidden state
   - 输出: logits (vocab_size = 248320)
   - 对比: logits 分布

### 2.3 最可能的根因假设

假设 1: PLE 表解析错误

IQ2_XS 使用 PLE（Piecewise Linear Encoding）量化。本引擎的 PLE 表读取可能与 Strata 不同：
- 表的位置（分片 00002）
- 表的格式（行优先 vs 列优先）
- 表的插值逻辑

假设 2: Token Embedding 量化

如果 token embedding 也是量化的（IQ2_XS），本引擎的反量化逻辑可能有 bug。

假设 3: GPU 计算精度

CUDA 内核的浮点精度问题：
- FP16 vs FP32 累积误差
- GDN 递推的数值稳定性
- QSA 注意力的 softmax 精度

---

## 3. 修复方案

### 3.1 短期方案：使用 Strata 引擎

决策: 既然 Strata 引擎输出正确，先用 Strata 引擎作为生产后端。

实施:
1. 完善 StrataBackend 的 EngineDriver 接线
2. 解决 Strata 引擎启动问题（内存/配置）
3. 性能优化（批处理、KV 缓存）

优点:
- 立即可用，输出正确
- 性能已优化（80-100 tok/s）

缺点:
- 依赖外部引擎
- 无法深度优化

### 3.2 长期方案：修复自研引擎

步骤:

1. Token Embedding 对拍
   ```cpp
   // test_token_embed_parity.cpp
   // 对比 Strata 和自研的 token embedding 输出
   ```

2. PLE 表对拍
   ```cpp
   // test_ple_parity.cpp
   // 对比 PLE 表的读取和插值
   ```

3. 逐层对拍
   ```cpp
   // test_layer_parity.cpp
   // 对比每一层的输出（GDN → QSA → MoE）
   ```

4. Logits 对拍
   ```cpp
   // test_logits_parity.cpp
   // 对比最终 logits 分布
   ```

---

## 4. 多 GPU 协同设计

### 4.1 双 GPU Dense 层切分

现状: T8 设计了双卡切分，但 `dense_exec_cuda()` 只返回单 GPU 函数指针。

设计:

```cpp
// DenseExec 扩展
struct DenseExec {
    // 单 GPU 路径（向后兼容）
    RunGdnLayerFn run_gdn;
    RunQsaLayerFn run_qsa;
    
    // 双 GPU 路径（新增）
    RunGdnLayerFn run_gdn_gpu0;
    RunGdnLayerFn run_gdn_gpu1;
    RunQsaLayerFn run_qsa_gpu0;
    RunQsaLayerFn run_qsa_gpu1;
    
    // 层切分配置
    int gpu0_layers;  // GPU 0 处理的层数（如 24）
    int gpu1_layers;  // GPU 1 处理的层数（如 24）
};
```

实现:

```cpp
// dense_exec_cuda.cu
DenseExec dense_exec_cuda_dual(int gpu0, int gpu1) {
    DenseExec e;
    
    // GPU 0: 层 0-23
    e.run_gdn_gpu0 = [gpu0 (...) {
        cudaSetDevice(gpu0);
        run_gdn_layer_cuda(...);
    };
    
    // GPU 1: 层 24-47
    e.run_gdn_gpu1 = [gpu1 (...) {
        cudaSetDevice(gpu1);
        run_gdn_layer_cuda(...);
    };
    
    e.gpu0_layers = 24;
    e.gpu1_layers = 24;
    
    return e;
}
```

Step Loop 修改:

```cpp
// step_loop.cpp
for (int l = 0; l < n_layers; ++l) {
    if (l < dense_exec.gpu0_layers) {
        // GPU 0 处理
        dense_exec.run_gdn_gpu0(...);
    } else {
        // GPU 1 处理
        dense_exec.run_gdn_gpu1(...);
    }
}
```

### 4.2 MoE 专家 GPU 1 卸载

设计: GPU 1 闲置时，可以卸载部分 MoE 专家到 GPU 1。

```cpp
// ExpertConfig
struct ExpertConfig {
    int cpu_experts;      // CPU 处理的专家数
    int gpu0_experts;     // GPU 0 处理的专家数
    int gpu1_experts;     // GPU 1 处理的专家数
    
    // 专家路由
    std::function<int(int expert_id)> get_device;
};
```

### 4.3 iGPU 接入（可选）

现状: P3a 完成三档着色器，P3b 取消。

重新评估: 如果双 GPU 已满载，iGPU 可以处理轻量任务：
- Token embedding 查找
- LM head 投影
- 采样器计算

设计:

```cpp
// IgpuExec
struct IgpuExec {
    // iGPU 处理的任务
    RunTokenEmbedFn run_token_embed_igpu;
    RunLmHeadFn run_lm_head_igpu;
};
```

---

## 5. 实施计划

### Phase 1: 乱码调查（1-2 天）

1. T1: 纯 CPU 路径测试（排除 GPU 精度问题）
2. T2: Token embedding 对拍（vs Strata）
3. T3: PLE 表对拍（vs Strata）
4. T4: 逐层对拍（定位 bug 层）
5. T5: 修复 + 验证

### Phase 2: 双 GPU 切分（2-3 天）

1. T6: DenseExec 扩展（双 GPU 函数指针）
2. T7: Step loop 修改（层切分逻辑）
3. T8: 权重双加载（GPU 0 + GPU 1 各一份）
4. T9: 性能测试 + 调优

### Phase 3: iGPU 接入（可选，1-2 天）

1. T10: iGPU 任务分配设计
2. T11: Vulkan 接线恢复
3. T12: 性能测试

---

## 6. 验收标准

### 6.1 乱码修复

- [ ] Token embedding 与 Strata 逐位一致（或误差 < 1e-5）
- [ ] PLE 表读取与 Strata 逐位一致
- [ ] 逐层输出与 Strata 误差 < 1e-3
- [ ] 最终 logits 与 Strata 误差 < 1e-2
- [ ] 输出文本连贯可读

### 6.2 双 GPU 协同

- [ ] GPU 0 + GPU 1 利用率均 > 20%
- [ ] 推理速度比单 GPU 提升 > 1.5x
- [ ] 输出质量不下降（乱码问题已修复）

### 6.3 全协同（含 iGPU）

- [ ] GPU 0 + GPU 1 + iGPU + CPU 全部工作
- [ ] 推理速度比双 GPU 提升 > 1.2x
- [ ] 显存占用均衡

---

## 7. 风险与缓解

### 7.1 乱码修复风险

风险: 量化格式解析差异难以定位  
缓解: 先用 Strata 引擎，自研引擎慢慢调试

### 7.2 双 GPU 切分风险

风险: 层间同步开销大，性能提升不明显  
缓解: 测试不同切分比例（24/24, 20/28, 28/20）

### 7.3 iGPU 接入风险

风险: iGPU 性能太弱，收益不大  
缓解: 只做轻量任务（embedding/LM head）

---

## 8. 决策记录

### 8.1 乱码修复优先级

决策: 先修复乱码，再做多 GPU 协同  
理由: 乱码是 blocker，多 GPU 是优化

### 8.2 Strata 引擎 vs 自研引擎

决策: 生产环境先用 Strata，自研引擎并行开发  
理由: Strata 已验证，自研需要时间调试

### 8.3 iGPU 是否接入

决策: 暂缓，等双 GPU 切分完成后再评估  
理由: iGPU 收益不确定，优先级低

---

## 9. 附录

### 9.1 相关文件

- `src/serve/native_backend.cpp`
- `src/serve/facade.cpp`
- `src/dense/dense_exec_cuda.cu`
- `src/dense/step_loop.cpp`

### 9.2 参考

- ADR-004: Consumer Dual GPU No Tensor Parallel
- ADR-019: iGPU not for expert GEMV
- Strata 引擎源码: `~/projects/ref_repos/strata`

---

下一步: 开始 Phase 1 乱码调查，从纯 CPU 路径测试开始。
