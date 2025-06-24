# Negative Flip Analysis: Object Detection Model Comparison

Mathematical analysis of "negative flips" to quantify performance degradation between object detection models on the COCO dataset.

## 🎯 What is Negative Flip?

**Negative Flip**: when an object is detected by the baseline (Model A) but missed by the new model (Model B)
- **Location Negative Flip (LNF)**: Objects correctly detected by Model A but missed by Model B
- **Classification Negative Flip (CNF)**: Objects detected by both models, but misclassified by Model B
- **Total Negative Flip (TNF)**: Combined failures across both categories

## 📐 Rates
``` 
LNF_rate = LNF_total / N_total
CNF_rate = CNF_total / N_loc  
Flip_difference = CNF_rate - LNF_rate
``` 
Where:

- *N_total*: Total ground truth objects
- *N_loc*: Objects localized by both models
- *Flip Difference*: Indicates whether problems are primarily localization (< 0) or classification (> 0) related

## 📦 Dataset Setup

You can download COCO by running: 
``` 
python src/training-yolo/dowload_coco.py
``` 


## 📊 Experiment 1: Architectural Evolution (YOLOv8n vs YOLOv11n)
**Research Question:** How do architectural improvements affect detection performance?]

### Results Summary
Analysis on 36,335 COCO validation objects (IoU threshold: 0.5)

| Metric | Value | Interpretation |
|------|-------|-------------|
| LNF Rate | 5.10% | YOLOv11n misses 5.1% of objects that YOLOv8n detects |
| CNF Rate  | 0.53% | Among jointly detected objects, YOLOv11n misclassifies 0.53% |
| TNF Rate | 5.35% | Overall percentage of negative flips (either location or classification) |
| Flip Difference | -4.85% | Localization issues dominate over classification issues |

#### Key Statistics

- Total Objects: 36,335
- Both Models Detected objects: 17,474 (48.1%)
- Location Negative Flips: 1,853
- Classification Negative Flips: 92


## 📊 Experiment 2: Training Data Scale Impact (Half COCO vs Full COCO)
**Research Question:** How does training data size affect detection capabilities?

Both models are trained from scratch using the same **YOLOv11n** architecture.

### Training Curves 
![Training curves](./plots/train_loss.png)

### Validation Curves
![Validation curves](./plots/validation_loss.png)



### Standard metrics
| Model     | mAp   | mAp50 | mAp75 | Precision  | Recall | 
|-----------|-------|-------|-------|------------|--------|
| YOLOv11n - Half COCO | 4.00% | 7.02% | 3.91% | 32.33%      | 9.92%  
| YOLOv11n - Full COCO | 17.42% | 26.59% | 39.11% | 39.63%     | 27.22%

### Results Summary

| Metric | Value | Interpretation |
|------|-------|-------------|
| LNF Rate | 0.64% | Half-COCO model finds 233 objects Full-COCO model misses |
| CNF Rate  | 0.81% | Full-COCO model's classification error rate|
| TNF Rate | 0.73% | Overall percentage of negative flips (either location or classification) |
| Flip Difference | -0.55% | Slight localization advantage for half-trained model |

### Key Statistics

- Total Objects: 36,335
- Both Models Detect: 4,179 (11.5% - much lower overlap)
- Location Negative Flips: 233
- Classification Negative Flips: 34