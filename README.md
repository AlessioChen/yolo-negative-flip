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
**Research Question:** How do architectural improvements affect detection performance?

### Standard metrics
| Model     | mAp   | mAp50 | mAp75 | Precision  | Recall | 
|-----------|-------|-------|-------|------------|--------|
| YOLOv8n  - Pre trained | 37.14% | 52.11% | 40.39% | 63.41%      | 47.43%
| YOLOv11n - Pre trained | 39.25% | 54.86% | 42.73% | 65.29%      | 50.43%


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

# 📊 Experiment 3: Knowledge Distillation (KD) Impact (YOLOv8n Teacher → YOLOv11n Student)
**Research Question**: Can knowledge distillation from an older architecture improve newer model performance and reduce negative flips?

## Methology
### KD Setup:
- **Teacher Model**: YOLOv8n (pre-trained)
- **Student Model**: YOLOv11n (trained from scratch with KD)
- Two Distillation Types Tested:
    1. **Feature-based KD**: L2 distillation on intermediate features
    2. **Response-based KD**: KL divergence on detection head outputs

### Feature-based Distillation
**Layer Mapping**: 
- YOLOv8 layers [15, 18, 21] → YOLOv11 layers [16, 19, 22]
- Targets P3, P4, P5 multi-scale features (small, medium, large objects)
**Loss Function**: $$\text{Standard YOLO Loss} + \alpha \times \text{L2(student features, teacher feataures)}$$

### Response-based Distillation

- **Distillation Target**: Detection head outputs (bbox + classification logits)
- **Loss Function**: $$ \text{Standard YOLO Loss} + \alpha \times  \text{KL divergence(student logits, teacher logits)}$$
- **Strategy** : Focal weighting with confidence thresholding on teacher predictions

### Common Training Setup:

- **Training**: 30 epochs on full COCO dataset
- **Image size**: 640×640
- **Optimizer**: SGD with standard YOLO settings


### Standard metrics
| Model     | mAp   | mAp50 | mAp75 | Precision  | Recall | 
|-----------|-------|-------|-------|------------|--------|
| YOLOv8n  - Pre trained (Teacher) | 37.14% | 52.11% | 40.39% | 63.41%      | 47.43%
| YOLOv11n - Feature KD from YOLOv8n | 36.53%| 51.36% | 39.37%| 63.17%     | 47.01%
| YOLOv11n - Response KD from YOLOv8n | 36.64% | 51.53% | 39.69% | 63.63%      | 47.17%
| YOLOv11n - Pre trained | 39.25% | 54.86% | 42.73% | 65.29%      | 50.43%


### Negative Flip Analysis Results
Comparison: YOLOv8n (Teacher/Baseline) vs YOLOv11n with Knowledge Distillation (Student)

**Response based KD Results** 
| Metric | Value | Interpretation |
|------|-------|-------------|
| LNF Rate | 7.88% | YOLOv11n Response KD misses 7.88%  of objects that YOLOv8n detects |
| CNF Rate  | 0.54% | Among jointly detected objects, YOLOv11n-Response-KD misclassifies 0.54% |
| TNF Rate | 8.12% | Overall percentage of negative flips (either location or classification) |
| Flip Difference | -7.63% | Localization differences dominate |



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
