from typing import Dict, List
from ..utils.iou import compute_iou
from ..models.detector import get_ground_truth_objects
from pycocotools.coco import COCO

def find_matching_detections(detections: List[Dict], gt_objects: List[Dict], iou_threshold: float = 0.5) -> Dict:
    """
    Find matching detections for ground truth objects based on IoU threshold.
    
    Args:
        detections: List of detections from model
        gt_objects: List of ground truth objects
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dict: Dictionary mapping ground truth indices to their best matching detection
    """
    matches = {}

    for gt_idx, gt_obj in enumerate(gt_objects):
        best_match = None 
        best_confidence = -1 

        for det in detections:
            iou = compute_iou(det['bbox'], gt_obj['bbox'])

            if iou >= iou_threshold and det['confidence'] > best_confidence:
                best_match = det
                best_confidence = det['confidence']

        matches[gt_idx] = best_match
                
    return matches

def analyze_negative_flips(results_v1: Dict, results_v2: Dict, coco: COCO, iou_threshold: float = 0.5) -> Dict:
    """
    Analyze negative flips between two model versions.
    
    Args:
        results_v1: Detection results from first YOLO model 
        results_v2: Detection results from second YOLO model 
        coco: COCO dataset instance
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dict: Analysis results including flip statistics and details
    """
    N_total = 0   # Total number of ground truth objects
    N_loc = 0     # Objects localized by both models

    LNF_total = 0  # Location negative flips
    CNF_total = 0  # Classification negative flips 
    BNF_total = 0  # Both negative flips 
    TNF_total = 0  # Total Negative Flips (either location or classification)

    flip_details = []

    for img_id in list(results_v1.keys()):
        gt_objects = get_ground_truth_objects(img_id, coco)
        N_total += len(gt_objects)

        detections_v1 = results_v1[img_id]
        detections_v2 = results_v2[img_id]

        matches_v1 = find_matching_detections(detections_v1, gt_objects, iou_threshold)
        matches_v2 = find_matching_detections(detections_v2, gt_objects, iou_threshold)

        for gt_idx, gt_obj in enumerate(gt_objects):
            d1 = matches_v1[gt_idx]
            d2 = matches_v2[gt_idx]
            
            # Location Negative Flip: LNF_{i,g} = 1 if d1 ≠ ∅ and d2 = ∅
            LNF_i_g = 1 if (d1 is not None and d2 is None) else 0 

            # Classification Negative Flip: CNF_{i,g} = 1 if both detected but v8 correct, v11 wrong
            CNF_i_g = 0 
            if d1 is not None and d2 is not None: 
                N_loc += 1  # Count objects localized by both models 
                if d1['class'] == gt_obj['class'] and d2['class'] != gt_obj['class']:
                    CNF_i_g = 1

            BNF_i_g = LNF_i_g * CNF_i_g  
            TNF_i_g = 1 if (LNF_i_g == 1 or CNF_i_g == 1) else 0

            LNF_total += LNF_i_g
            CNF_total += CNF_i_g
            BNF_total += BNF_i_g
            TNF_total += TNF_i_g

            if LNF_i_g == 1 or CNF_i_g == 1:
                flip_details.append({
                    'image_id': img_id,
                    'gt_class': gt_obj['class'],
                    'gt_bbox': gt_obj['bbox'],
                    'LNF': LNF_i_g,
                    'CNF': CNF_i_g,
                    'TNF': TNF_i_g,
                    'v1_detected': d1 is not None,
                    'v2_detected': d2 is not None,
                    'v1_class': d1['class'] if d1 else None,
                    'v2_class': d2['class'] if d2 else None,
                    'v1_confidence': d1['confidence'] if d1 else None,
                    'v2_confidence': d2['confidence'] if d2 else None
                })

    # Calculate rates
    LNF_rate = LNF_total / N_total if N_total > 0 else 0
    CNF_rate_standard = CNF_total / N_loc if N_loc > 0 else 0  # Standard: denominator = N_loc
    CNF_rate_common_denom = CNF_total / N_total if N_total > 0 else 0  # Common denominator for subtraction
    TNF_rate = TNF_total / N_total if N_total > 0 else 0

    # Compute the difference (using common denominator)
    flip_difference = CNF_rate_common_denom - LNF_rate

    return {
        'summary': {
            'N_total': N_total,
            'N_loc': N_loc,
            'LNF_total': LNF_total,
            'CNF_total': CNF_total,
            'BNF_total': BNF_total,
            'TNF_total': TNF_total,
            'LNF_rate': LNF_rate,
            'CNF_rate_standard': CNF_rate_standard,
            'CNF_rate_common_denom': CNF_rate_common_denom,
            'TNF_rate': TNF_rate,
            'flip_difference': flip_difference,
            'iou_threshold': iou_threshold
        },
        # 'flip_details': flip_details
    } 