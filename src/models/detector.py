import os
from typing import Dict, List
from ultralytics import YOLO
from pycocotools.coco import COCO
from tqdm import tqdm

def run_inference(model: YOLO, image_ids: list[int], coco: COCO, coco_val_path: str) -> Dict:
    """
    Run inference on a set of images using a YOLO model.
    
    Args:
        model: YOLO model instance
        image_ids: List of COCO image IDs
        coco: COCO dataset instance
        coco_val_path: Path to COCO validation images
        
    Returns:
        Dict: Dictionary mapping image IDs to their detections
    """
    results = {}
    for img_id in tqdm(image_ids, desc="Processing images"):
        
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(f"{coco_val_path}/{img_info['file_name']}")
    
        try:
            prediction = model(img_path, verbose=False)[0]
            detections = []
            if prediction.boxes is not None: 
                boxes = prediction.boxes.xyxy.cpu().numpy()
                confidences = prediction.boxes.conf.cpu().numpy()
                classes = prediction.boxes.cls.cpu().numpy()
    
                for i in range(len(boxes)):
                    detections.append({
                        'bbox': boxes[i].tolist(), 
                        'confidence': float(confidences[i]),
                        'class': int(classes[i])
                    })
    
            results[img_id] = detections
                
        except Exception as e: 
            print(f"Error processing image {img_id}: {e}")

    return results

def get_ground_truth_objects(img_id: int, coco: COCO) -> List[Dict]:
    """
    Get ground truth objects for a given image ID.
    
    Args:
        img_id: COCO image ID
        coco: COCO dataset instance
        
    Returns:
        List[Dict]: List of ground truth objects with their bounding boxes and classes
    """
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    gt_objects = []

    for ann in anns: 
        if ann['iscrowd'] == 0:  # Only consider non-crowd annotations
            # Convert from [x,y,w,h] to [x1,y1,x2,y2]
            x, y, w, h = ann['bbox']
            bbox = [x, y, x + w, y + h]

            gt_objects.append({
                'bbox': bbox, 
                'class': ann['category_id'] - 1,  # COCO classes are 1-indexed, convert to 0-indexed
                'area': ann['area']
            })
    return gt_objects 