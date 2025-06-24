from ultralytics import YOLO
import os
import yaml
import tempfile 

def evaluate_model_performance(model, val_images_path, annotations_path):

    dataset_config = create_coco_dataset_config(val_images_path, annotations_path)
    
    try:
        results = model.val(
            data=dataset_config,
            verbose=False,
            save=False,
            plots=False
        )
        
        metrics = {
            'mAP': float(results.box.map) if results.box.map is not None else 0.0,
            'mAP50': float(results.box.map50) if results.box.map50 is not None else 0.0,
            'mAP75': float(results.box.map75) if results.box.map75 is not None else 0.0,
            'precision': float(results.box.mp) if results.box.mp is not None else 0.0,
            'recall': float(results.box.mr) if results.box.mr is not None else 0.0,
        }
        
        metrics['f1'] = 0.0
        
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        
     

        model_info = {
            'parameters': sum(p.numel() for p in model.model.parameters()) if hasattr(model, 'model') else 0,
            'model_path': str(model.ckpt_path) if hasattr(model, 'ckpt_path') else None
        }
        
        if model_info['model_path'] and os.path.exists(model_info['model_path']):
            model_info['model_size_mb'] = os.path.getsize(model_info['model_path']) / (1024 * 1024)
        
        return {
            'metrics': metrics,
            'model_info': model_info
        }
        
    except Exception as e:
        return {
            'model_name': model_name,
            'metrics': {
                'mAP': 0.0,
                'mAP50': 0.0,
                'mAP75': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            },
            'model_info': {'parameters': 0, 'model_size_mb': 0},
            'error': str(e)
        }
    
    finally:
        if os.path.exists(dataset_config):
            os.remove(dataset_config)

def create_coco_dataset_config(val_images_path, annotations_path):    
    coco_names = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
        27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
        32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
        36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
        56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
        61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
        67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
        72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
        77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }
    
    data_root = os.path.dirname(val_images_path)
    val_relative = os.path.relpath(val_images_path, data_root)
    
    config = {
        'path': data_root,
        'train': val_relative,  # Use val path for train too (we're only validating)
        'val': val_relative,
        'names': coco_names
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        temp_config_path = f.name
    
    return temp_config_path

if __name__ == "__main__":

    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)


    model_v1_path = config['models']['v1']
    model_v2_path = config['models']['v2']
    
    model_v1 = YOLO(model_v1_path)
    model_v2 = YOLO(model_v2_path)

    coco_ann_path = config['dataset']['annotations_path']
    coco_val_path = config['dataset']['images_path']

    performance_v1 = evaluate_model_performance(model_v1, coco_val_path, coco_ann_path)
    performance_v2 = evaluate_model_performance(model_v2, coco_val_path, coco_ann_path)

    print(performance_v1)

    print('----------------------')
    print(performance_v2)