# Negative Flip Analysis: Object Detection Model Comparison

This work introduces a **Negagive Flip Analysis** to quantify performance regression between object detection models beyond standard metrics such as mAP. 
All experiments are conducted on the COCO validation dataset using YOL-based detector.

## 🎯 What is Negative Flip?
A **Negative Flip** occurs when an object is correctly detected by a baseline model (Model A) but missed by the new model (Model B).

I defined three types: 
- **Location Negative Flip (LNF)**: Objects correctly detected by Model A but missed by Model B
- **Classification Negative Flip (CNF)**: Objects detected by both models, but misclassified by Model B
- **Total Negative Flip (TNF)**: Combined failures across both categories

## 📐 Negative flip Rates
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
All experiments use: 
- COCO validation set
- IoU threshold = 0.5 
- 36,335 ground-truth objects


## 📊 Experiment 1: Architectural Evolution (YOLOv8n vs YOLOv11n)
**Research Question:** How do architectural improvements affect detection performance and negative flips?

### Standard metrics
| Model     | mAp   | mAp50 | mAp75 | Precision  | Recall | 
|-----------|-------|-------|-------|------------|--------|
| YOLOv8n  - Pre trained | 37.14% | 52.11% | 40.39% | 63.41%      | 47.43%
| YOLOv11n - Pre trained | **39.25%** | **54.86%** | **42.73%** | **65.29%** | **50.43%**

### Negative Flip Results 

| Metric | Value | Interpretation |
|------|-------|-------------|
| LNF Rate | 5.10% | YOLOv11n misses 5.1% of objects that YOLOv8n detects |
| CNF Rate  | 0.53% | Very few misclassifications among shared detections|
| TNF Rate | 5.35% | Overall negative flip rate |
| Flip Difference | **-4.85%** | Localization dominates |

#### Key Statistics

- Total Objects: **36,335**
- Both Models Detected objects: **17,474 (48.1%)**
- Location Negative Flips: **1,853**
- Classification Negative Flips: **92**


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
| YOLOv11n - Full COCO | **34.99%** | **49.41%** |  **38.09%** | **60.35%**     | **45.88%**

### Results Summary

| Metric | Value | Interpretation |
|------|-------|-------------|
| LNF Rate | 2.30% | Half-COCO model finds 836 objects Full-COCO model misses |
| CNF Rate  | 0.40% | Classification remains stable |
| TNF Rate | 2.46% | Overall negative flip rate  |
| Flip Difference | -2.14% | Localization dominates |

### Key Statistics

- Total Objects: **36,335**
- Both Models Detect: **14840  (40.84%)**
- Location Negative Flips (LNF): **836**
- Classification Negative Flips (CNF): **60**

# 📊 Experiment 3: Knowledge Distillation 
**Research Question**: Can knowledge distillation reduce negative flips when transferring from an older architecture?

## Methology
### KD Setup:
- **Teacher**: YOLOv8n (pre-trained)
- **Student**: YOLOv11n (trained from scratch)
- **Training**: 30 epochs on full COCO dataset
- **Image size**: 640×640
- **Optimizer**: SGD with standard YOLO settings

Two KD strategies were evaluated. 

### Feature-based KD
The objective is to encourage intermediate representation similarity between teacher and student 

**Layer**: 
- YOLOv8 layers [15, 18, 21] → YOLOv11 layers [16, 19, 22]
- Targets P3, P4, P5 multi-scale features (small, medium, large objects)

### Response-based KD
The objective is to transfer class probabilities structure from the teacher detection head to the student 

- **Target**: Detection head classification logits
- **Strategy** : Focal weighting with confidence thresholding on teacher predictions

### Common Training Setup:

- **Training**: 30 epochs on full COCO dataset
- **Image size**: 640×640
- **Optimizer**: SGD with standard YOLO settings


## Loss function details 

**Base detection Loss (YOLOv8 /YOLOv11)**

$$L_{YOLO} = L_{box} + L_{cls} + L_{dfl}$$

**Combined training objective with KD**

$$L_{Total} = L_{YOLO} + \alpha \times L_{KD}$$


### Response based KD loss 
The objective is to transfer class probabilities structure from the teacher detection head to the student 

$$
L_{KD}^{resp} = \frac{1}{L} \sum_{l=1}^{L} w_{focal}^l \cdot 
KL\Big(\text{softmax}(\frac{z_s^l}{T}), \text{softmax}(\frac{z_t^l}{T})\Big)
$$

where: 

- $$z_s, z_t$$, students and teacher classification logits 
- T: temperature scaling 
- L: detection head layers 
- $${w}_{focal}$$: confidence-aware weighting 


Distillation is applied only when the teacher is confident

$$p_t=\max (softmax(z_t)) > conf_thresh)$$

Focal Weighting: each location is reweighted

$$w_{focal} = (1 - p_t)^{\gamma}$$


### Feature based KD loss 
The objective is to encourage intermediate representation similarity between teacher and student 

$$L_{{KD}}^{{feat}} = \frac{1}{K} \sum_{k=1}^{K} \| F_s^{k} - F_t^{k} \|_2^2$$

where: 

- $$F_s^{k} - F_{t}^{k}$$: student and teacher feature maps 
- $$K$$: P3,P4,P3 feature levels 



### Standard metrics
| Model     | mAp   | mAp50 | mAp75 | Precision  | Recall | 
|-----------|-------|-------|-------|------------|--------|
| YOLOv8n  - Pre trained (Teacher) | 37.14% | 52.11% | 40.39% | 63.41%      | 47.43%
| YOLOv11n - Feature KD from YOLOv8n | 36.53%| 51.36% | 39.37%| 63.17%     | 47.01%
| YOLOv11n - Response KD from YOLOv8n | 36.64% | 51.53% | 39.69% | 63.63%      | 47.17%
| YOLOv11n - Pre trained | **39.25%** | **54.86%** | **42.73%** | **65.29%**      | **50.43%**

### Negative Flip Results (KS vs NO-KD)
Comparison: YOLOv8n (Teacher/Baseline) vs YOLOv11n with Knowledge Distillation (Student)

| Metric | Experiment 1 | Response KD | Feature KD |
|------|------------------|------------------|------------------|
| **LNF Rate** | 2.30%| **8.06% | 7.88% |
| **CNF Rate** |0.40%  | 0.49% | 0.54% |
| **TNF Rate** | 2.46% | 8.29% | 8.12% |
| **Flip Difference** | -2.14% | -7.84% | -7.63% |
| **Dominant Error Type** | Localizatio | Localization | Localization |


| KD increases localization negative flips by ~55% compared to the non-KD baseline, while classification consistency remains largely unchanged.

### Key Statistics Comparison

| Statistic | Experiment 1|  Response KD | Feature KD |
|---------|------------------|------------------|------------------|
| **Total Objects** | 36,335  | 36,335 | 36,335 |
| **Objects Detected by Both Models** |17,474 (48.1%) | 16,399 (45.12%) | 16,464 (45.2%) |
| **Location Negative Flips** |1,853 | **2,928** | **2,863** |
| **Classification Negative Flips**| 92 | 81 | 89 |

| KD reduces overlap between teacher and student detections, indicating degraded spatial generalization rather than semantic confusion.





# 📍  Conlusion 
## 1.Training Data Impact vs Universal Performance

The analysis reveals that maintaining the same architecture (YOLOv11n) while increasing training data from half to the complete COCO dataset significantly improves standard metrics performance. 
However, i can observe a considerable number of localization negative flips (836 objects, 2.3%), meaning the model trained with less data detects objects that the model trained with more data cannot find.

This suggests that **increasing training data does not lead to universal improvement across all objects, but rather causes the model to learn different strategies for object identification**. Different data volumes create distinct detection 
specializations rather than purely superior performance.

## 2.Localization vs Classification Challenges

From the experiments, i observe that when both models detect the same object, they almost always assign the same class (very low CNF rates). This indicates that:

- Object detection negative flips are concentrated more on localization than  classification
- Classification performance remains consistent across different models when objects are successfully detected
- The primary challenge in model evolution is preserving detection coverage, not improving classification accuracy

## 3.Knowledge Distillation & Negative Flip Analysis
Despite architectural advances in YOLO11n, i observe that KD from YOLOV8n does not reduce negative flips, and in fact it has increased localication failures, even though standard metrcs remain competive. 

