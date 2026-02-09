# Individual LoRA Evaluation Results

> 评测日期: 2026-02-09
> 
> 测评方式: 每个 LoRA 单独加载到 base model 上，在其对应的测试数据上进行推理

---

## Summary

| Model | LoRA Type | Overall Step Acc | Overall Episode Acc | Avg Step Acc | Avg Episode Acc |
|-------|-----------|------------------|---------------------|--------------|-----------------|
| InternVL2-2B | Category | 91.26% (94/103) | 91.54% (65/71) | 91.58% | 92.34% |
| InternVL2-2B | App | 89.20% (223/250) | 89.20% (223/250) | 88.93% | 88.93% |
| Qwen2-VL-7B | Category | 92.23% (95/103) | 92.96% (66/71) | 92.73% | 93.54% |
| Qwen2-VL-7B | App | 90.40% (226/250) | 90.40% (226/250) | 90.00% | 90.00% |

> **Overall Accuracy** = 总正确数 / 总样本数
> 
> **Avg Accuracy** = 各LoRA准确率的平均值

---

## InternVL2-2B Category LoRAs

| Category | Samples | Step Accuracy | Episode Accuracy |
|----------|---------|---------------|------------------|
| entertainment | 15 | 100.00% | 100.00% |
| lives | 21 | 80.95% | 81.25% |
| office | 19 | 89.47% | 93.33% |
| shopping | 33 | 87.50% | 87.10% |
| traveling | 15 | 100.00% | 100.00% |
| **Total** | **103** | **91.26%** | **91.54%** |

---

## InternVL2-2B App LoRAs

| App | Samples | Step Accuracy | Episode Accuracy |
|-----|---------|---------------|------------------|
| adidas | 20 | 50.00% | 50.00% |
| amazon | 15 | 100.00% | 100.00% |
| calendar | 20 | 95.00% | 95.00% |
| clock | 15 | 100.00% | 100.00% |
| decathlon | 20 | 80.00% | 80.00% |
| ebay | 15 | 100.00% | 100.00% |
| etsy | 20 | 75.00% | 75.00% |
| flipkart | 15 | 100.00% | 100.00% |
| gmail | 15 | 100.00% | 100.00% |
| google_drive | 20 | 95.00% | 95.00% |
| google_maps | 20 | 95.00% | 95.00% |
| kitchen_stories | 20 | 80.00% | 80.00% |
| reminder | 15 | 100.00% | 100.00% |
| youtube | 20 | 75.00% | 75.00% |
| **Total** | **250** | **89.20%** | **89.20%** |

---

## Qwen2-VL-7B Category LoRAs

| Category | Samples | Step Accuracy | Episode Accuracy |
|----------|---------|---------------|------------------|
| entertainment | 15 | 100.00% | 100.00% |
| lives | 21 | 85.71% | 87.50% |
| office | 19 | 84.21% | 86.67% |
| shopping | 33 | 93.75% | 93.55% |
| traveling | 15 | 100.00% | 100.00% |
| **Total** | **103** | **92.23%** | **92.96%** |

---

## Qwen2-VL-7B App LoRAs

| App | Samples | Step Accuracy | Episode Accuracy |
|-----|---------|---------------|------------------|
| adidas | 20 | 65.00% | 65.00% |
| amazon | 15 | 100.00% | 100.00% |
| calendar | 20 | 95.00% | 95.00% |
| clock | 15 | 100.00% | 100.00% |
| decathlon | 20 | 65.00% | 65.00% |
| ebay | 15 | 100.00% | 100.00% |
| etsy | 20 | 75.00% | 75.00% |
| flipkart | 15 | 100.00% | 100.00% |
| gmail | 15 | 100.00% | 100.00% |
| google_drive | 20 | 100.00% | 100.00% |
| google_maps | 20 | 90.00% | 90.00% |
| kitchen_stories | 20 | 80.00% | 80.00% |
| reminder | 15 | 100.00% | 100.00% |
| youtube | 20 | 90.00% | 90.00% |
| **Total** | **250** | **90.40%** | **90.40%** |

---

## Source Data

- `internvl2_category_20260209-194327/results_summary.csv`
- `internvl2_app_20260209-194409/results_summary.csv`
- `qwen2vl_category_20260209-194445/results_summary.csv`
- `qwen2vl_app_20260209-173228/results_summary.csv`
