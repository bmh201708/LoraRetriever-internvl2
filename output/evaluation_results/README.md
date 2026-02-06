# LoraRetriever Evaluation Summary (InternVL2-2B)

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
