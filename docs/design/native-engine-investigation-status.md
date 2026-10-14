# Native Engine 乱码调查状态报告

**日期**: 2026-10-14  
**状态**: 调查中

---

## 1. 已完成的调查

### 1.1 对比测试

| Backend | Build | GPU | 输出 | 结论 |
|---------|-------|-----|------|------|
| NativeBackend | CPU-only | 无 | 乱码 | 问题在底层 |
| NativeBackend | CUDA | GPU 0 | 乱码 | 问题在底层 |
| Inhouse | CUDA | GPU 0 | 乱码 | 问题在底层 |
| Strata | 参考 | GPU | 连贯（据 runbook） | 我们的管线有 bug |

### 1.2 GPU 状态

- **GPU 0**: ✅ 工作中（~27% 利用率，326 MiB）
- **GPU 1**: ❌ 闲置（0%）
- **iGPU**: ❌ 未接入（P3b 取消）
- **CPU**: ✅ 工作中（MoE + 嵌入 + LM head）

### 1.3 测试限制

- **夹具模型**: 词表太小（248320 vs 256），多字符提示失败
- **真模型**: 内存需求大，启动困难（OOM 风险）
- **Strata 引擎**: 启动配置复杂，需要 --spec/--mtp 等参数

---

## 2. 乱码根因假设

### 假设 1: PLE 表解析错误（最可能）

**证据**:
- IQ2_XS 使用 PLE 量化
- 我们的 PLE 表读取逻辑可能与 Strata 不同
- PLE 表在分片 00002 中

**验证方法**:
```cpp
// test_ple_parity.cpp
// 对比 Strata 和自研的 PLE 表读取
```

### 假设 2: Token Embedding 量化

**证据**:
- Token embedding 可能也是量化的
- 反量化逻辑可能有 bug

**验证方法**:
```cpp
// test_token_embed_parity.cpp
// 对比 token 1 的 embedding 输出
```

### 假设 3: GPU 计算精度

**证据**:
- CUDA 内核的浮点精度问题
- FP16 vs FP32 累积误差

**验证方法**:
- 纯 CPU 路径测试（已尝试，启动失败）
- 对比 GPU vs CPU 路径的 logits

### 假设 4: Chat Template 渲染

**证据**:
- 模板渲染的 prompt 格式可能不同
- 特殊 token 处理可能不一致

**验证方法**:
```cpp
// test_chat_template_parity.cpp
// 对比渲染后的 prompt 文本
```

---

## 3. 下一步计划

### Phase 1A: 隔离问题层（优先级最高）

1. **T1: 纯 CPU 路径测试**
   - 用更小的模型或更少的内存
   - 排除 GPU 精度问题

2. **T2: Token Embedding 对拍**
   - 写测试：读取 token 1 的 embedding
   - 对比 Strata 的输出

3. **T3: PLE 表对拍**
   - 写测试：读取 PLE 表的一行
   - 对比 Strata 的输出

4. **T4: 逐层 Logits 对拍**
   - 写测试：输入相同 token，对比 layer 48 输出
   - 定位 bug 在哪一层

### Phase 1B: 修复

5. **T5: 修复 PLE 表解析**（如果假设 1 正确）
6. **T6: 修复 Token Embedding**（如果假设 2 正确）
7. **T7: 修复 GPU 内核**（如果假设 3 正确）

---

## 4. 决策建议

### 短期（本周）

**决策**: 先用 Strata 引擎作为生产后端

**理由**:
- Strata 已验证，输出正确
- 自研引擎调试需要时间
- 不阻塞用户使用

**实施**:
- 完善 StrataBackend 的 EngineDriver 接线
- 解决 Strata 引擎启动问题

### 中期（下周）

**决策**: 并行调查自研引擎乱码问题

**理由**:
- 自研引擎可以深度优化
- 长期依赖外部引擎有风险

**实施**:
- 按 Phase 1A 计划逐层对拍
- 定位并修复 bug

### 长期（本月）

**决策**: 双 GPU 切分 + iGPU 接入

**理由**:
- 最大化硬件利用率
- 提升推理吞吐

**实施**:
- 等乱码修复后开始
- 按设计文档实施

---

## 5. 风险评估

| 风险 | 影响 | 缓解 |
|------|------|------|
| 乱码难以定位 | 高 | 先用 Strata，自研慢慢调 |
| 双 GPU 性能提升不明显 | 中 | 测试不同切分比例 |
| iGPU 收益不大 | 低 | 只做轻量任务 |

---

## 6. 需要的资源

1. **内存**: 真模型需要 ~40GB，当前 61GB 可用，但 IDE cgroup 可能被 OOM
2. **时间**: 乱码调查预计 1-2 天
3. **工具**: 需要 Strata 引擎的中间输出（logits/embedding）做对比

---

**下一步**: 等待用户决策——先修乱码还是先用 Strata？
