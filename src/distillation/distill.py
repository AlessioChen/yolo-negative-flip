import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from comet_ml import Experiment
from datetime import datetime

import os
import logging

os.environ["COMET_DISPLAY_SUMMARY_LEVEL"] = "0"
os.environ["COMET_LOGGING_FILE_LEVEL"] = "WARNING"


def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"kd_training_{timestamp}.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


class KDLoss(v8DetectionLoss):
    def __init__(
        self, model, teacher_model=None, alpha=0.7, conf_thresh=0.1, gamma=1.0, T=2.0
    ):
        super().__init__(model)
        self.model = model
        self.teacher_model = teacher_model

        self.alpha = alpha  # Weight for KD loss
        self.conf_thresh = conf_thresh
        self.gamma = gamma
        self.T = T  # Temperature for softmax
        self.epsilon = 1e-6

        self.reg_max = getattr(model.model[-1], "reg_max", 16)  # usually 16 for v8/v11
        self.nc = model.nc
        self.batch_counter = 0
        self.epoch_counter = 0

        if self.teacher_model:
            self.teacher_model.eval()
            self.teacher_model.to(next(model.parameters()).device)
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def __call__(self, preds, batch):
        # 1. Standard YOLO Loss (Box, Class, DFL)
        loss_vec, loss_items = super().__call__(preds, batch)
        loss = loss_vec.sum()  # scalar YOLO loss

        if not self.model.training or self.teacher_model is None:
            zero = loss_items.new_zeros(1)
            loss_items = torch.cat([loss_items, zero])
            return loss, loss_items

        # 2. Get Teacher Logits
        with torch.no_grad():
            device = preds[0].device
            if next(self.teacher_model.parameters()).device != device:
                self.teacher_model.to(device)

            # The student 'preds' during training is a list of [P3, P4, P5]
            # Each shape: [Batch, 64 + nc, H, W]
            input_tensor = batch["img"]
            self.teacher_model.to(preds[0].device)

            # Forward pass teacher
            teacher_outputs = self.teacher_model(
                input_tensor.to(next(self.teacher_model.parameters()).dtype)
            )

            teacher_preds = (
                teacher_outputs[1]
                if isinstance(teacher_outputs, tuple)
                else teacher_outputs
            )

        # 3. Calculate Distillation Loss
        dist_loss = self.compute_response_distill_loss(preds, teacher_preds)

        # 4. Scale and Combine
        total_loss = loss + (self.alpha * dist_loss)

        if self.batch_counter % 50 == 0:
            dist_loss_float = (
                dist_loss.item()
                if isinstance(dist_loss, torch.Tensor)
                else float(dist_loss)
            )
            total_loss_float = (
                total_loss.item()
                if isinstance(total_loss, torch.Tensor)
                else float(total_loss)
            )
            loss_float = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

            logger.info(
                f"[Epoch {self.epoch_counter}, Batch {self.batch_counter}] "
                f"Distillation Loss: {dist_loss_float:.4f} (alpha={self.alpha})"
            )
            logger.info(
                f"[Epoch {self.epoch_counter}, Batch {self.batch_counter}] "
                f"Combined Loss: {total_loss_float:.4f} "
                f"(YOLO: {loss_float:.4f} + KD: {self.alpha * dist_loss_float:.4f})"
            )

        # Update logs
        dist_loss_val = dist_loss.detach()
        loss_items = torch.cat([loss_items, dist_loss_val.view(1)])

        self.batch_counter += 1
        return total_loss, loss_items

    def compute_response_distill_loss(self, student_preds, teacher_preds):
        total_cls_loss = 0.0
        valid_layers = 0

        for i, (s_pred, t_pred) in enumerate(zip(student_preds, teacher_preds)):
            # Match shapes if necessary
            if s_pred.shape != t_pred.shape:
                logger.info(
                    f"Skipping KD at layer {i} due to shape mismatch: "
                    f"{s_pred.shape} vs {t_pred.shape}"
                )
                continue

            B, C, H, W = s_pred.shape
            split_idx = 4 * self.reg_max

            # CLS logits
            s_cls = s_pred[:, split_idx:]
            t_cls = t_pred[:, split_idx:]

            # Teacher confidence (CLS)
            # [B, nc, H, W] → softmax over classes
            t_prob = F.softmax(t_cls / self.T, dim=1)
            t_conf, _ = t_prob.max(dim=1, keepdim=True)  # [B,1,H,W]

            # Posive congruent mask
            mask = (t_conf > self.conf_thresh).float()
            mask_ratio = mask.mean().item()

            if mask.sum() < 1:
                logger.info(f"[KD] Layer {i}: no valid KD locations")
                continue

            # Focal weight
            focal_weight = (1.0 - t_conf).pow(self.gamma)
            weight = mask * focal_weight

            # Classification KD (softmax)
            s_log_softmax = F.log_softmax(s_cls / self.T, dim=1)
            t_softmax = F.softmax(t_cls / self.T, dim=1)
            cls_loss = F.kl_div(s_log_softmax, t_softmax, reduction="none")

            cls_weight = weight.expand_as(s_cls)
            cls_loss = (cls_loss * cls_weight).sum() / (cls_weight.sum() + self.epsilon)

            total_cls_loss += cls_loss
            valid_layers += 1

            if self.batch_counter % 50 == 0:
                logger.info(
                    f"[KD][Layer {i}] "
                    f"cls_kd={cls_loss.item():.4f}, "
                    f"mask_ratio={mask_ratio:.4f}, "
                    f"t_conf_mean={t_conf.mean().item():.4f}, "
                    f"focal_mean={focal_weight.mean().item():.4f}"
                )

        if valid_layers == 0:
            logger.info("No valid layers for distillation loss calculation")
            return torch.tensor(0.0, device=student_preds[0].device)

        return (total_cls_loss) / valid_layers


class KD_Model(DetectionModel):
    def __init__(self, cfg, teacher_weights, **kwargs):
        super().__init__(cfg, **kwargs)
        self.teacher_weights = teacher_weights

    def init_criterion(self):
        if self.teacher_weights:
            print(f"Loading teacher model for Response KD: {self.teacher_weights}")
            teacher = YOLO(self.teacher_weights).model
        else:
            teacher = None
        return KDLoss(self, teacher_model=teacher)


class KDTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        self.teacher_weights = overrides.pop("teacher", None)
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=False):
        model = KD_Model(
            cfg,
            teacher_weights=self.teacher_weights,
            nc=self.data["nc"],
            verbose=verbose,
        )
        if weights:
            model.load(weights)
        return model

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = ["box_loss", "cls_loss", "dfl_loss", "dist_loss"]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip([f"{prefix}/{x}" for x in keys], loss_items))
        else:
            return keys


if __name__ == "__main__":
    logger = setup_logging()

    teacher_model_path = "yolov8n.pt"
    student_model_path = "yolo11n.pt"

    kd_alpha = 0.7
    kd_conf_thresh = 0.1
    kd_gamma = 1.0
    kd_temperature = 2.0

    dataset = "coco"
    epochs = 30
    imgsz = 640

    experiment_name = (
        f"KD_Y11n_from_Y8n_"
        f"a{kd_alpha}_"
        f"c{kd_conf_thresh}_"
        f"g{kd_gamma}_"
        f"T{kd_temperature}_"
        f"e{epochs}_"
        f"{dataset}"
    )

    experiment = Experiment(
        project_name="YOLO-Negative-flip",
    )

    experiment.set_name(experiment_name)

    hyperparams = {
        # KD hyperparameters
        "kd_alpha": kd_alpha,
        "kd_conf_thresh": kd_conf_thresh,
        "kd_gamma": kd_gamma,
        "kd_temperature": kd_temperature,
        # Training hyperparameters
        "epochs": epochs,
        "imgsz": imgsz,
        "dataset": dataset,
        # Model information
        "teacher_model": "yolov8n",
        "student_model": "yolo11n",
        # Loss configuration
        "loss_type": "response_kd",
        "distillation_type": "focal_positive_congruent",
        "kd_components": "cls",
        "mask_type": "confidence_threshold",
    }
    experiment.log_parameters(hyperparams)

    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Hyperparameters: {hyperparams}")

    trainer = KDTrainer(
        overrides={
            "model": student_model_path,
            "data": f"{dataset}.yaml",
            "epochs": epochs,
            "imgsz": imgsz,
            "device": "cuda",
            "teacher": teacher_model_path,
        }
    )

    trainer.train()
