from typing import List

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box in format [x1, y1, x2, y2]
        box2: Second bounding box in format [x1, y1, x2, y2]
        
    Returns:
        float: IoU score between 0 and 1
    """
    if len(box1) != 4 or len(box2) != 4:
        raise ValueError("Box format not supported")

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    intersection = w * h

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - intersection

    iou = intersection / union if union > 0 else 0
    return iou 