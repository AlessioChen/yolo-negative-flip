from comet_ml import Experiment
from ultralytics import YOLO
import os
from dotenv import load_dotenv

load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY")

def train_yolo(dataset_yaml: str, experiment: Experiment, epochs: int = 10, batch: int = 16, save_path: str = None):
    model = YOLO('yolo11n.yaml')  # build model from scratch
    model.model.nc = 80


    experiment.log_parameter("epochs", epochs)
    experiment.log_parameter("batch_size", batch)
    experiment.log_parameter("img_size", 640)
    experiment.log_parameter("dataset_yaml", dataset_yaml)

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
    epochs = 10
    batch = 16

    half_dataset_yaml = 'coco_half.yml'
    full_dataset_yaml = 'coco_full.yml'

    print("Training on half COCO dataset...")
    exp_half = Experiment(
        api_key=COMET_API_KEY,
        project_name="yolo11n",
        auto_output_logging="simple"
    )
    exp_half.set_name("YOLOv11n - Half COCO")
    
    results_half = train_yolo(
        half_dataset_yaml,
        experiment=exp_half,
        epochs=epochs,
        batch=batch,
        save_path="yolo11n_half.pt"
    )

    print("Training on full COCO dataset...")
    exp_full = Experiment(
        api_key=COMET_API_KEY,
        project_name="yolo11n",
        auto_output_logging="simple"
    )
    exp_full.set_name("YOLOv11n - Full COCO")
    results_full = train_yolo(
        full_dataset_yaml,
        experiment=exp_full,
        epochs=epochs,
        batch=batch,
        save_path="yolo11n_full.pt"
    )

    print("mAP@0.5 on half dataset:", results_half.metrics.get('mAP_0.5'))
    print("mAP@0.5 on full dataset:", results_full.metrics.get('mAP_0.5'))


if __name__ == "__main__":
    main()