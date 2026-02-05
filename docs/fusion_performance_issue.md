# Fusion模式性能问题分析与解决方案

## 问题描述

在使用 `--merge_method fusion` 进行推理时，第一个样本可以正常完成，但从第二个样本开始，推理变得极其缓慢（等待10分钟以上仍未完成）。

## 问题根源分析

### 当前Fusion实现流程

```
每个样本推理时:
1. FusionComposer.compose()
   ↓
2. model.add_weighted_adapter(adapters, weights, adapter_name="fused_N")
   ↓
3. SwiftModel.add_weighted_adapter() [base.py:453-543]
   - 创建临时 LoraModel 对象
   - 调用 PEFT 的 lora_model.add_weighted_adapter() 合并LoRA参数
   - 将新adapter注册到 self.adapters["fused_N"]
   - 调用 self.set_active_adapters("fused_N")  ← 性能瓶颈！
   ↓
4. set_active_adapters() [base.py:629-649]
   - 激活新的 fused_N adapter
   - 停用所有其他adapter
   ↓
5. 对每个需要停用的adapter调用 deactivate_adapter()
   - 遍历模型所有LoRA层，设置activation状态
```

### 性能瓶颈

`set_active_adapters()` 的实现（base.py:643-647）：

```python
for adapter_name in (adapter_names & set(self.adapters.keys())):
    self.activate_adapter(adapter_name)

for adapter_name in (set(self.adapters.keys()) - adapter_names):
    self.deactivate_adapter(adapter_name, offload)  # 停用所有其他adapter
```

问题：
1. 每次创建新的 `fused_N` adapter后，需要停用所有其他adapter
2. 需要停用的adapter包括：14个原始LoRA + 之前创建的所有fused adapter (fused_1, fused_2, ...)
3. 每个adapter的停用操作需要遍历模型的所有LoRA层
4. 随着样本数量增加，累积的fused adapter越来越多，停用操作越来越慢

### 时间复杂度分析

假设：
- 原始LoRA数量: M = 14
- 已处理样本数: N
- 模型LoRA层数: L

每处理一个新样本的时间复杂度：O((M + N) * L)

随着N增加，性能线性下降。

## 解决方案

### 方案：只创建一次fused adapter，后续直接更新参数

核心思路：
1. 第一次调用时，使用 `add_weighted_adapter` 创建 fused adapter
2. 后续调用时，直接遍历LoRA层，更新 fused adapter 的参数（lora_A, lora_B）
3. 避免重复调用 `set_active_adapters`，从而避免停用大量adapter的开销

优点：
- 时间复杂度降为 O(L)，与样本数量N无关
- 只需遍历一次LoRA层更新参数

实现要点：
```python
# 第一次：创建fused adapter
model.add_weighted_adapter(adapters, weights, adapter_name="fused")

# 后续：直接更新fused adapter的参数
for module in model.modules():
    if isinstance(module, LoraLayer) and "fused" in module.lora_A:
        # 计算新的融合参数
        new_lora_A = weighted_sum([module.lora_A[name] for name in adapters], weights)
        new_lora_B = weighted_sum([module.lora_B[name] for name in adapters], weights)
        # 直接更新参数
        module.lora_A["fused"].data.copy_(new_lora_A)
        module.lora_B["fused"].data.copy_(new_lora_B)
```

## 相关代码位置

- `lora_retriever/composition.py`: FusionComposer 类
- `swift/tuners/base.py:453-543`: SwiftModel.add_weighted_adapter()
- `swift/tuners/base.py:629-649`: SwiftModel.set_active_adapters()
- `swift/tuners/lora.py:159-165`: LoRA.activate_adapter()

## 参考

论文中的Fusion定义（Section 4.2.2）：
```
Θ_fusion = (1/k) * Σ_{j=1}^{k} Θ_j
```

即将top-k个LoRA的参数做加权平均，得到一个融合后的LoRA参数。
