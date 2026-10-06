# Memotion 7k Confusion Matrix & Error Analysis Summary

Evaluated on test split across seeds [0, 1, 2].

### Text-only (BERT)
- **Mean Accuracy:** `0.4571`
- **Mean Macro-F1:** `0.3094`

| True \ Predicted | Negative (0) | Neutral (1) | Positive (2) | Total Support |
|---|---|---|---|---|
| **Negative** | 32 (19.8%) | 40 (24.7%) | 90 (55.6%) | 162 |
| **Neutral** | 90 (13.8%) | 174 (26.6%) | 390 (59.6%) | 654 |
| **Positive** | 156 (12.1%) | 374 (29.1%) | 754 (58.7%) | 1284 |


### Image-only (ResNet-50)
- **Mean Accuracy:** `0.4386`
- **Mean Macro-F1:** `0.3081`

| True \ Predicted | Negative (0) | Neutral (1) | Positive (2) | Total Support |
|---|---|---|---|---|
| **Negative** | 19 (11.7%) | 44 (27.2%) | 99 (61.1%) | 162 |
| **Neutral** | 98 (15.0%) | 157 (24.0%) | 399 (61.0%) | 654 |
| **Positive** | 203 (15.8%) | 336 (26.2%) | 745 (58.0%) | 1284 |


### Late Concat Fusion
- **Mean Accuracy:** `0.4267`
- **Mean Macro-F1:** `0.3369`

| True \ Predicted | Negative (0) | Neutral (1) | Positive (2) | Total Support |
|---|---|---|---|---|
| **Negative** | 27 (16.7%) | 66 (40.7%) | 69 (42.6%) | 162 |
| **Neutral** | 79 (12.1%) | 265 (40.5%) | 310 (47.4%) | 654 |
| **Positive** | 130 (10.1%) | 550 (42.8%) | 604 (47.0%) | 1284 |


### Product Fusion
- **Mean Accuracy:** `0.4086`
- **Mean Macro-F1:** `0.3210`

| True \ Predicted | Negative (0) | Neutral (1) | Positive (2) | Total Support |
|---|---|---|---|---|
| **Negative** | 24 (14.8%) | 73 (45.1%) | 65 (40.1%) | 162 |
| **Neutral** | 79 (12.1%) | 274 (41.9%) | 301 (46.0%) | 654 |
| **Positive** | 133 (10.4%) | 591 (46.0%) | 560 (43.6%) | 1284 |


### Cross-Attention Fusion
- **Mean Accuracy:** `0.4676`
- **Mean Macro-F1:** `0.3269`

| True \ Predicted | Negative (0) | Neutral (1) | Positive (2) | Total Support |
|---|---|---|---|---|
| **Negative** | 15 (9.3%) | 58 (35.8%) | 89 (54.9%) | 162 |
| **Neutral** | 44 (6.7%) | 225 (34.4%) | 385 (58.9%) | 654 |
| **Positive** | 67 (5.2%) | 475 (37.0%) | 742 (57.8%) | 1284 |

