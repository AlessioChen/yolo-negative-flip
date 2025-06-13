from comet_ml import Experiment
from ultralytics import YOLO
import os
import argparse
from dotenv import load_dotenv

load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY")

def train_yolo(dataset_yaml: str, experiment: Experiment, epochs: int = 10, batch: int = 16, save_path: str = None):
    model = YOLO('yolo11n.yaml')  # build model from scratch
    model.model.nc = 80

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=640,
        device="cuda"
    )

    if save_path:
        model.save(save_path)
        experiment.log_model("trained_model", save_path)

    mAP_05 = results.metrics.get("mAP_0.5")
    experiment.log_metric("mAP@0.5", mAP_05)

    experiment.end()
    return results


def main():
    
    parser = argparse.ArgumentParser(description="Train YOLOv11n on COCO datasets")
    parser.add_argument('--dataset', choices=['half', 'full'], required=True, help='Choose dataset: half or full')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')

    args = parser.parse_args()
    
    if args.dataset == 'half':
        dataset_yaml = 'coco_half.yml'
        experiment_name = "YOLOv11n - Half COCO"
        save_path = "yolo11n_half.pt"
    else:
        dataset_yaml = 'coco_full.yml'
        experiment_name = "YOLOv11n - Full COCO"
        save_path = "yolo11n_full.pt"

    print(f"Training on {args.dataset} COCO dataset...")
    
    experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name="yolo11n",
        auto_output_logging="simple"
    )
    experiment.set_name(experiment_name)

    results = train_yolo(
        dataset_yaml,
        experiment=experiment,
        epochs=args.epochs,
        batch=args.batch,
        save_path=save_path
    )

    print(f"mAP@0.5 on {args.dataset} dataset:", results.metrics.get('mAP_0.5'))


if __name__ == "__main__":
    main()