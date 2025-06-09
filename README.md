# Negative Flip Analysis: YOLOv8 vs YOLOv11 


Mathematical Analysis of "negative flips" between YOLOv8 nano and YOLOv11 nano models on the COCO dataset to assess the impact of architectual changes.

## Setup Dataset 

1. Download COCO validation set
- Images: [val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
- Annotations: [2017 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

## Results

Analysis performed on COCO validation set with IoU threshold of 0.5:

| Rate | Value | Description |
|------|-------|-------------|
| LNF Rate | 5.10% | Percentage of objects where YOLOv8 detects but YOLOv11 misses |
| CNF Rate (Standard) | 0.53% | Percentage of jointly detected objects where YOLOv8 classifies correctly but YOLOv11 misclassifies |
| CNF Rate (Common Denominator) | 0.25% | Percentage of all objects where YOLOv8 classifies correctly but YOLOv11 misclassifies |
| TNF Rate | 5.35% | Overall percentage of negative flips (either location or classification) |
| Flip Difference | -4.85% | Difference between CNF and LNF rates, indicating relative impact of classification vs location errors |
