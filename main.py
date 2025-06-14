import json
import numpy as np
from ultralytics import YOLO
from pycocotools.coco import COCO
from src.models.detector import run_inference
from src.analysis.negative_flips import analyze_negative_flips
import os
import yaml


def main():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
        
    iou_threshold = config['inference']['iou_threshold']
    output_dir = config['output']['results_dir'] 
    output_file = config['output']['results_file']
    model_v1_path = config['models']['v1']
    model_v2_path = config['models']['v2']
    coco_ann_path = config['dataset']['annotations_path']
    coco_val_path = config['dataset']['images_path']


    model_v1 = YOLO(model_v1_path)
    model_v2 = YOLO(model_v2_path)


    coco = COCO(coco_ann_path)
    image_ids = coco.getImgIds()

    result_v1 = run_inference(model_v1, image_ids, coco, coco_val_path)
    result_v2 = run_inference(model_v2, image_ids, coco, coco_val_path)
    results = analyze_negative_flips(result_v1, result_v2, coco, iou_threshold)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)    

    with open(output_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {
            'summary': {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) 
                       for k, v in results['summary'].items()},
        }
        json.dump(json_results, f, indent=2)

if __name__ == "__main__":
    main() 