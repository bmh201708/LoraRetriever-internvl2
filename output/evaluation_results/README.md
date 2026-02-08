# LoraRetriever Evaluation Summary

---

# InternVL2-2B

**Model:** InternVL2-2B | **Top-K:** 3

---

## Mixture Mode

**Date:** 2026-02-06 21:55:49

### App-level Results (14 App LoRAs)

| App | Step-level Accuracy | Episode-level Accuracy |
|-----|---------------------|------------------------|
| adidas | 60.00% | 60.00% |
| amazon | 93.33% | 93.33% |
| calendar | 90.00% | 90.00% |
| clock | 93.33% | 93.33% |
| decathlon | 60.00% | 60.00% |
| ebay | 86.67% | 86.67% |
| etsy | 80.00% | 80.00% |
| flipkart | 66.67% | 66.67% |
| gmail | 100.00% | 100.00% |
| google_drive | 100.00% | 100.00% |
| google_maps | 90.00% | 90.00% |
| kitchen_stories | 65.00% | 65.00% |
| reminder | 100.00% | 100.00% |
| youtube | 70.00% | 70.00% |
| **Average** | **82.50%** | **82.50%** |
| **OVERALL** | **81.60%** | **81.78%** |

### Category-level Results (5 Category LoRAs)

| Category | Step-level Accuracy | Episode-level Accuracy |
|----------|---------------------|------------------------|
| entertainment | 80.00% | 76.92% |
| lives | 47.62% | 50.00% |
| office | 68.42% | 73.33% |
| shopping | 87.50% | 87.10% |
| traveling | 86.67% | 83.33% |
| **Average** | **74.04%** | **74.14%** |
| **OVERALL** | **74.51%** | **75.58%** |

---

## Fusion Mode

**Date:** 2026-02-06 22:31:21

### App-level Results (14 App LoRAs)

| App | Step-level Accuracy | Episode-level Accuracy |
|-----|---------------------|------------------------|
| adidas | 65.00% | 65.00% |
| amazon | 73.33% | 73.33% |
| calendar | 90.00% | 90.00% |
| clock | 93.33% | 93.33% |
| decathlon | 60.00% | 60.00% |
| ebay | 86.67% | 86.67% |
| etsy | 80.00% | 80.00% |
| flipkart | 66.67% | 66.67% |
| gmail | 100.00% | 100.00% |
| google_drive | 95.00% | 95.00% |
| google_maps | 90.00% | 90.00% |
| kitchen_stories | 60.00% | 60.00% |
| reminder | 100.00% | 100.00% |
| youtube | 75.00% | 75.00% |
| **Average** | **81.07%** | **81.07%** |
| **OVERALL** | **80.40%** | **80.57%** |

### Category-level Results (5 Category LoRAs)

| Category | Step-level Accuracy | Episode-level Accuracy |
|----------|---------------------|------------------------|
| entertainment | 73.33% | 69.23% |
| lives | 42.86% | 43.75% |
| office | 63.16% | 66.67% |
| shopping | 84.38% | 83.87% |
| traveling | 86.67% | 83.33% |
| **Average** | **70.08%** | **69.37%** |
| **OVERALL** | **70.59%** | **72.09%** |

---

## Overall Comparison (Mixture vs Fusion)

| Level | Merge Method | Step-level Accuracy | Episode-level Accuracy |
|-------|-------------|---------------------|------------------------|
| App-level OVERALL | Mixture | **81.60%** | **81.78%** |
| App-level OVERALL | Fusion | 80.40% | 80.57% |
| Category-level OVERALL | Mixture | **74.51%** | **75.58%** |
| Category-level OVERALL | Fusion | 70.59% | 72.09% |

---

# Qwen2-VL-7B-Instruct

**Model:** Qwen2-VL-7B-Instruct | **Top-K:** 3

---

## Mixture Mode

**Date:** 2026-02-08 01:16:20

### App-level Results (14 App LoRAs)

| App | Step-level Accuracy | Episode-level Accuracy |
|-----|---------------------|------------------------|
| adidas | 65.00% | 65.00% |
| amazon | 93.33% | 93.33% |
| calendar | 100.00% | 100.00% |
| clock | 100.00% | 100.00% |
| decathlon | 55.00% | 55.00% |
| ebay | 93.33% | 93.33% |
| etsy | 90.00% | 90.00% |
| flipkart | 86.67% | 86.67% |
| gmail | 100.00% | 100.00% |
| google_drive | 100.00% | 100.00% |
| google_maps | 95.00% | 95.00% |
| kitchen_stories | 75.00% | 75.00% |
| reminder | 100.00% | 100.00% |
| youtube | 80.00% | 80.00% |
| **Average** | **88.09%** | **88.09%** |
| **OVERALL** | **87.20%** | **87.45%** |

### Category-level Results (5 Category LoRAs)

| Category | Step-level Accuracy | Episode-level Accuracy |
|----------|---------------------|------------------------|
| entertainment | 80.00% | 76.92% |
| lives | 90.48% | 87.50% |
| office | 84.21% | 86.67% |
| shopping | 96.88% | 96.77% |
| traveling | 93.33% | 91.67% |
| **Average** | **88.98%** | **87.91%** |
| **OVERALL** | **90.20%** | **89.53%** |

---

## Fusion Mode

**Date:** 2026-02-08 01:37:17

### App-level Results (14 App LoRAs)

| App | Step-level Accuracy | Episode-level Accuracy |
|-----|---------------------|------------------------|
| adidas | 60.00% | 60.00% |
| amazon | 86.67% | 86.67% |
| calendar | 100.00% | 100.00% |
| clock | 100.00% | 100.00% |
| decathlon | 50.00% | 50.00% |
| ebay | 93.33% | 93.33% |
| etsy | 95.00% | 95.00% |
| flipkart | 86.67% | 86.67% |
| gmail | 100.00% | 100.00% |
| google_drive | 95.00% | 95.00% |
| google_maps | 90.00% | 90.00% |
| kitchen_stories | 70.00% | 70.00% |
| reminder | 100.00% | 100.00% |
| youtube | 85.00% | 85.00% |
| **Average** | **86.55%** | **86.55%** |
| **OVERALL** | **85.60%** | **85.83%** |

### Category-level Results (5 Category LoRAs)

| Category | Step-level Accuracy | Episode-level Accuracy |
|----------|---------------------|------------------------|
| entertainment | 73.33% | 69.23% |
| lives | 90.48% | 87.50% |
| office | 78.95% | 86.67% |
| shopping | 96.88% | 96.77% |
| traveling | 93.33% | 91.67% |
| **Average** | **86.59%** | **86.37%** |
| **OVERALL** | **88.24%** | **88.37%** |

---

## Overall Comparison (Mixture vs Fusion)

| Level | Merge Method | Step-level Accuracy | Episode-level Accuracy |
|-------|-------------|---------------------|------------------------|
| App-level OVERALL | Mixture | **87.20%** | **87.45%** |
| App-level OVERALL | Fusion | 85.60% | 85.83% |
| Category-level OVERALL | Mixture | **90.20%** | **89.53%** |
| Category-level OVERALL | Fusion | 88.24% | 88.37% |

---

# Cross-Model Comparison (InternVL2-2B vs Qwen2-VL-7B)

| Model | Level | Merge Method | Step-level Accuracy | Episode-level Accuracy |
|-------|-------|-------------|---------------------|------------------------|
| InternVL2-2B | App-level | Mixture | 81.60% | 81.78% |
| InternVL2-2B | App-level | Fusion | 80.40% | 80.57% |
| InternVL2-2B | Category-level | Mixture | 74.51% | 75.58% |
| InternVL2-2B | Category-level | Fusion | 70.59% | 72.09% |
| Qwen2-VL-7B | App-level | Mixture | **87.20%** | **87.45%** |
| Qwen2-VL-7B | App-level | Fusion | 85.60% | 85.83% |
| Qwen2-VL-7B | Category-level | Mixture | **90.20%** | **89.53%** |
| Qwen2-VL-7B | Category-level | Fusion | 88.24% | 88.37% |
