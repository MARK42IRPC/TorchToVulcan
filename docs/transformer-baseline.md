# Transformer Vulkan Baseline

## 目标

这一阶段的工程目标是让一个静态、FP32、无控制流的 Transformer block
能够被编译成 TTV 0.1 线性包，并在真实 Vulkan 设备上完成一次前向计算。
正确性、可诊断性和可复现的验证证据优先于性能。首版允许每个输出元素在
shader 内部串行执行 reduction 或矩阵内积，但不能在包内静默回退到 CPU。

目标 block 的最小形态是：

```text
X
 |\
 | MatMul/常量投影
 |  + bias
 |  LayerNormalization
 |  Softmax 或 attention score
 |  MatMul/常量投影
 +---------------------------------> Y
```

这是编译器能力的验收样例，不是 Aemeath 的完整 TTS pipeline。Aemeath 的
动态序列、KV-cache、量化权重、音频图和 host loop 在这个 block 通过后再
逐项接入。

## 阶段与验收

### P0：静态批量 MatMul

范围：FP32、contiguous、静态 shape、输入 rank 至少为 2。支持
`[batch..., M, K] @ [batch..., K, N]`，batch 维度按 ONNX trailing
broadcast 规则对齐；输出为广播后的 batch shape 加 `[M, N]`。

验收：

- 计划器能生成明确的 batch offset 和 `[M, N]` 索引；
- 非法 K、输出 shape、batch broadcast 和 rank 输入给出节点级诊断；
- shader 能通过 SPIR-V 编译和验证；
- TTV 包能在 Vulkan 设备上执行，并与 ONNX Runtime CPU 参考逐元素比较；
- 二维 MatMul 的现有包兼容性保持不变。

当前状态：已完成。二维和静态 batch FP32 MatMul 已通过计划器、SPIR-V、真实
Vulkan 包执行以及 ONNX Runtime 差分验证。

### P1：Reduction 基础

范围：FP32 `ReduceMean`、静态 shape、常量 axes、`keepdims=0/1`，先覆盖
连续 reduction axes 和空 axes（按 ONNX 语义归约全部轴）。axes 必须在编译
期可取得，不能把未知运行时控制参数伪装成静态属性。

验收：

- reduction 的输出 shape 和元素计数由规范化 IR 验证；
- shader 对每个输出元素负责完整归约，处理 `keepdims` 两种形式；
- axes 缺失、重复、越界、类型错误和动态 axes 均有明确阻止原因；
- 至少覆盖标量输出、最后一维和多维归约的 CPU/Vulkan 差分测试。

当前状态：已完成第一项 `ReduceMean`。axes 可来自 opset attribute 或小型
initializer；initializer 只用于编译期选择，不作为 shader descriptor。

### P2：Softmax 与 LayerNormalization

范围：FP32、contiguous、静态 rank/shape、任意合法 axis。Softmax 首版在
一个 invocation 内完成 max、exp、sum 和归一化；LayerNormalization 首版
只输出 `Y`，支持从 `axis` 开始的 trailing normalized shape、scale/bias
以及有限 epsilon。

验收：

- 数值参考使用稳定的 max-subtraction 公式；
- LayerNorm 的 mean/variance 计算和 epsilon 语义与 ONNX Runtime 对齐；
- 缺失 scale/bias、shape 不匹配、非 contiguous layout 和额外输出均阻止；
- SPIR-V、真实 Vulkan 包执行和差分证书全部通过。

当前状态：`Softmax` 与 `LayerNormalization` 的 FP32 首版已通过真实 Vulkan
包执行和 device verification；仍不包含额外 mean/inv_std 输出。

### P3：静态 Transformer block

范围：将 P0-P2 组合成一个没有 `If`/`Loop`/`Scan`、没有动态 shape、没有
量化输入的完整 block。测试至少包含 `[batch, sequence, hidden]` 的输入、
投影权重和 bias，并同时覆盖 batch broadcast 的投影权重形式。

验收：

- 一个 ONNX 图可一次编译为线性 TTV 包；
- 包内所有运行时 dispatch 都有 Vulkan descriptor/push-constant 记录；
- 端到端输出同时通过 ONNX Runtime 和真实 Vulkan 设备校验；
- 失败时报告首个不支持节点及其 shape/type/attribute 原因；
- 不为通过测试临时加入模型专用算子或 CPU 分支。

当前状态：已完成一个静态 block 的包执行验收，包含两次 batch MatMul、bias
broadcast、LayerNormalization 和 Softmax；Vulkan 输出同时与 NumPy 和
ONNX Runtime 参考比较。下一步是把端到端样例纳入长期 Transformer fixture，
并继续处理 subprogram/state 契约。

### P4：TTV subprogram、host loop 与 KV-cache

范围：扩展包格式支持命名 subprogram、显式 profile、持久 state tensor、
host-driven loop 和小型 token/stop flag 回读。先由 host 驱动 autoregressive
迭代，设备端只负责 tensor 计算。

验收：

- 同一 Vulkan device/context 可以重复调用 subprogram；
- KV-cache 的 shape、layout、生命周期和更新边界写入包契约；
- loop 的退出条件、最大迭代次数和 host/device 同步点可审计；
- 旧的 `linear` TTV 0.1 包仍能被 runtime 读取。

### P5：量化与模型集成

范围：先支持编译期 FP32 反量化的 INT8/INT4 权重，再实现原生整数 kernel。
规范化 IR 必须保留 storage dtype、scale、zero point、axis、block size 和
accumulation dtype。Aemeath 只有在通用 contracts、subprogram 和验证链路
完成后才作为集成里程碑。

验收：

- INT8 per-tensor/per-channel 和 INT4 block metadata 不丢失；
- 反量化结果与独立 CPU 参考一致；
- `MatMulInteger`、`MatMulNBits` 等 vendor/operator-specific 路径有独立
  capability 记录，不冒充普通 FP32 MatMul；
- 完成至少一个真实 Aemeath 子图的 Vulkan 端到端证书。

## 通用 ONNX 覆盖规则

编译器按以下五级记录支持状态：semantic、normalized、lowered、compiled、
device verified。只有最后一级才能称为 Vulkan 可执行。每个 operator 的
支持声明必须同时写明 opset、dtype、shape/profile、layout、设备限制和
验证容差。

静态 TTV 0.1 的共同前提是：根图拓扑线性、shape 在编译时确定、tensor 为
显式声明的 contiguous buffer/view，并且所有输入和 initializer 都能在包
契约中物化。未知 shape、未绑定 profile、控制流、sequence、外部自定义域和
缺少 kernel 的节点必须让编译失败，不能生成部分包。

## 当前明确不支持

- 任意运行时动态 shape 或未经 `ShapeProfile` 特化的符号维度；
- `If`、`Loop`、`Scan` 和 sequence 类型的包内执行；
- FP16、BF16、INT8/INT4 原生计算；
- KV-cache、host-driven loop 和多 subprogram package；
- Conv、pooling、TopK、Gather、音频变换以及 Aemeath 专用 custom op 的
  Vulkan device verification；
- 以性能为理由的 fusion、arena 复用、异步流水和 autotuning。

这些限制是 support matrix 的一部分。新增算子必须先补语义/shape/type
测试，再补 kernel、SPIR-V 和 Vulkan/参考差分，不通过其中任一级就保持为
明确的 unsupported。
