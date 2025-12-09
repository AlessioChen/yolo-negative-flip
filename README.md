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
| YOLOv11n - Pre trained | 39.25% | 54.86% | 42.73% | 65.29%      | 50.43 
| YOLOv11n - Half COCO | 31.20% | 44.71% | 33.80% | 56.66%      | 41.60%  
| YOLOv11n - Full COCO | 34.99% | 49.41% |  38.09% | 60.35%     | 45.88%
| YOLOv11n - Distilled  from Yolov8n | 30.47% | 44.93% | 32.87% | 60.58% | 41.34 % 

### Results Summary

| Metric | Value | Interpretation |
|------|-------|-------------|
| LNF Rate | 2.30% | Half-COCO model finds 836 objects Full-COCO model misses |
| CNF Rate  | 0.40% | Full-COCO model's classification error rate|
| TNF Rate | 2.46% | Overall percentage of negative flips (either location or classification) |
| Flip Difference | -2.14% | Localization differences dominate |

### Key Statistics

- Total Objects: 36,335
- Both Models Detect: 14840  (40.84%)
- Location Negative Flips (LNF): 836
- Classification Negative Flips (CNF): 60


## 📊 Experiment 3: Knowledge Distillation (YOLOv8n → YOLOv11n)

**Research Question:** Does distilling a YOLOv8n teacher into a YOLOv11n student reduce negative flips and improve transfer of localization/classification knowledge?

This experiment compares:

Teacher: YOLOv8n

Student: YOLOv11n distilled using feature and logit-based KD (custom KD loss)


| Metric              | Value                | Interpretation                                                         |
| ------------------- | -------------------- | ---------------------------------------------------------------------- |
| **LNF Rate**        | **8.92%**            | Student (YOLOv11n KD) *misses 8.9%* of objects detected by the teacher |
| **CNF Rate**        | **0.56%** (standard) | Among jointly detected objects, student misclassifies ~0.6%            |
| **TNF Rate**        | **9.17%**            | Overall degradation relative to teacher detections                     |
| **Flip Difference** | **-8.67%**           | Large dominance of localization errors                                 |


The distillation attempt **failed** to transfer the teacher's localization capabilities. Instead of combining the best of both, the student became confused, performing worse than if it had just ignored the teacher.

- YOLOv8 and YOLOv11 have different internal architectures (C2f vs C3k2 blocks).
- Channel X in the teacher might not semantically map to Channel X in the student. Forcing them to match via KL Divergence might be destructive. 

# 📍  Conlusion 
## 1.Training Data Impact vs Universal Performance

The analysis reveals that maintaining the same architecture (YOLOv11n) while increasing training data from half to the complete COCO dataset significantly improves standard metrics performance. 
However, we can observe a considerable number of localization negative flips (836 objects, 2.3%), meaning the model trained with less data detects objects that the model trained with more data cannot find.

This suggests that **increasing training data does not lead to universal improvement across all objects, but rather causes the model to learn different strategies for object identification**. Different data volumes create distinct detection 
specializations rather than purely superior performance.

## 2.Localization vs Classification Challenges

From both experiments, we observe that when both models detect the same object, they almost always assign the same class (very low CNF rates). This indicates that:

- Object detection negative flips are concentrated more on localization than  classification
- Classification performance remains consistent across different models when objects are successfully detected
- The primary challenge in model evolution is preserving detection coverage, not improving classification accuracy
