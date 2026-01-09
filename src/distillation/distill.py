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


class FeatureExtractor(nn.Module):
    """Helper class to extract intermediate features from YOLO models"""

    def __init__(self, model, layer_indices):
        super().__init__()
        self.model = model
        self.layer_indices = layer_indices
        self.features = {}
        self.hooks = []

        for idx in layer_indices:
            layer = self.model.model[idx]
            hook = layer.register_forward_hook(self._make_hook_fn(idx))
            self.hooks.append(hook)

    def _make_hook_fn(self, layer_idx):
        """Create a closure that captures the layer index"""

        def hook_fn(module, input, output):
            # Store feature with layer index as key
            self.features[layer_idx] = output.detach()  # Detach to save memory

        return hook_fn

    def get_features(self):
        """Return features in order of layer indices"""
        return [self.features[idx] for idx in self.layer_indices]

    def clear_features(self):
        """Clear stored features"""
        self.features.clear()

    def remove_hooks(self):
        """Remove all hooks (call during cleanup)"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class KDLoss(v8DetectionLoss):
    def __init__(
        self,
        model,
        teacher_model=None,
        alpha=0.7,
        conf_thresh=0.1,
        gamma=1.0,
        T=2.0,
        kd_type="response",
        teacher_feature_layers=None,
        student_feature_layers=None,
        feature_beta=1.0,
    ):
        super().__init__(model)
        self.model = model
        self.teacher_model = teacher_model
        self.kd_type = kd_type

        self.alpha = alpha  # Weight for KD loss
        self.conf_thresh = conf_thresh
        self.gamma = gamma
        self.T = T  # Temperature for softmax
        self.feature_beta = feature_beta  # Weight for feature loss
        self.epsilon = 1e-6

        self.reg_max = getattr(model.model[-1], "reg_max", 16)
        self.nc = model.nc
        self.batch_counter = 0
        self.epoch_counter = 0

        self.teacher_feature_layers = teacher_feature_layers
        self.student_feature_layers = student_feature_layers

        if self.kd_type == "feature" and self.teacher_model:
            self.teacher_extractor = FeatureExtractor(
                self.teacher_model, teacher_feature_layers or [15, 18, 21]
            )
            self.student_extractor = FeatureExtractor(
                self.model, student_feature_layers or [16, 19, 22]
            )

            if teacher_feature_layers and student_feature_layers:
                if len(teacher_feature_layers) != len(student_feature_layers):
                    raise ValueError(
                        f"Teacher and student layer counts must match. "
                        f"Got teacher: {len(teacher_feature_layers)}, student: {len(student_feature_layers)}"
                    )

        if self.teacher_model:
            self.teacher_model.eval()
            self.teacher_model.to(next(model.parameters()).device)
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        logger.info(f"KDLoss initialized with kd_type='{self.kd_type}'")
        if self.kd_type == "feature":
            logger.info(f"Teacher layers: {teacher_feature_layers}")
            logger.info(f"Student layers: {student_feature_layers}")

    def __call__(self, preds, batch):
        # 1. Standard YOLO Loss (Box, Class, DFL)
        loss_vec, loss_items = super().__call__(preds, batch)
        loss = loss_vec.sum()  # scalar YOLO loss

        if not self.model.training or self.teacher_model is None:
            zero = loss_items.new_zeros(1)
            loss_items = torch.cat([loss_items, zero])
            return loss, loss_items

        device = preds[0].device
        if next(self.teacher_model.parameters()).device != device:
            self.teacher_model.to(device)

        # 2. Calculate Distillation Loss based on type
        if self.kd_type == "response":
            dist_loss = self._compute_response_distillation(preds, batch)
        elif self.kd_type == "feature":
            dist_loss = self._compute_feature_distillation(batch)
        else:
            raise ValueError(f"Unknown kd_type: {self.kd_type}")

        # 3. Scale and Combine
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
                f"Distillation Loss ({self.kd_type}): {dist_loss_float:.4f} (alpha={self.alpha})"
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

    def _compute_response_distillation(self, student_preds, batch):
        with torch.no_grad():
            input_tensor = batch["img"]
            teacher_outputs = self.teacher_model(
                input_tensor.to(next(self.teacher_model.parameters()).dtype)
            )
            teacher_preds = (
                teacher_outputs[1]
                if isinstance(teacher_outputs, tuple)
                else teacher_outputs
            )

        return self.compute_response_distill_loss(student_preds, teacher_preds)

    def _compute_feature_distillation(self, batch):
        """Feature-based L2 distillation with separate teacher/student layers"""
        input_tensor = batch["img"]

        with torch.no_grad():
            self.teacher_extractor.clear_features()  # Clear previous features
            _ = self.teacher_model(
                input_tensor.to(next(self.teacher_model.parameters()).dtype)
            )
            teacher_features = self.teacher_extractor.get_features()

        # Student features are already captured during the main forward pass
        # Just retrieve them from the hooks
        student_features = self.student_extractor.get_features()

        # Clear for next batch
        self.student_extractor.clear_features()

        return self.compute_feature_distill_loss(student_features, teacher_features)

    def compute_feature_distill_loss(self, student_features, teacher_features):
        """
        Compute L2 loss between student and teacher features
        """
        if len(student_features) != len(teacher_features):
            logger.warning(
                f"Feature count mismatch: student={len(student_features)}, "
                f"teacher={len(teacher_features)}"
            )
            return torch.tensor(0.0, device=student_features[0].device)

        total_loss = 0.0
        valid_layers = 0

        for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            # Get layer indices for logging
            teacher_idx = (
                self.teacher_feature_layers[i] if self.teacher_feature_layers else i
            )
            student_idx = (
                self.student_feature_layers[i] if self.student_feature_layers else i
            )

            # Compute L2 loss
            loss = F.mse_loss(s_feat, t_feat.detach())

            total_loss += loss * self.feature_beta
            valid_layers += 1

            if self.batch_counter % 50 == 0:
                logger.info(
                    f"[Feature KD][Map {i}: T_L{teacher_idx}→S_L{student_idx}] "
                    f"L2: {loss.item():.6f}, "
                    f"S_mean: {s_feat.mean().item():.4f}, T_mean: {t_feat.mean().item():.4f}, "
                    f"S_std: {s_feat.std().item():.4f}, T_std: {t_feat.std().item():.4f}"
                )

        if valid_layers == 0:
            logger.warning("No valid layers for feature distillation")
            return torch.tensor(0.0, device=student_features[0].device)

        return total_loss / valid_layers

    def compute_response_distill_loss(self, student_preds, teacher_preds):
        """Original response-based distillation loss"""
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
            t_prob = F.softmax(t_cls / self.T, dim=1)
            t_conf, _ = t_prob.max(dim=1, keepdim=True)  # [B,1,H,W]

            # Positive congruent mask
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
                    f"[Response KD][Layer {i}] "
                    f"cls_kd={cls_loss.item():.4f}, "
                    f"mask_ratio={mask_ratio:.4f}, "
                    f"t_conf_mean={t_conf.mean().item():.4f}, "
                    f"focal_mean={focal_weight.mean().item():.4f}"
                )

        if valid_layers == 0:
            logger.info("No valid layers for distillation loss calculation")
            return torch.tensor(0.0, device=student_preds[0].device)

        return total_cls_loss / valid_layers


class KD_Model(DetectionModel):
    def __init__(
        self,
        cfg,
        teacher_weights,
        kd_type="response",
        teacher_feature_layers=None,
        student_feature_layers=None,
        feature_beta=1.0,
        **kwargs,
    ):
        super().__init__(cfg, **kwargs)
        self.teacher_weights = teacher_weights
        self.kd_type = kd_type
        self.teacher_feature_layers = teacher_feature_layers
        self.student_feature_layers = student_feature_layers
        self.feature_beta = feature_beta

    def init_criterion(self):
        if self.teacher_weights:
            print(
                f"Loading teacher model for {self.kd_type.upper()} KD: {self.teacher_weights}"
            )
            if self.kd_type == "feature":
                print(f"Teacher feature layers: {self.teacher_feature_layers}")
                print(f"Student feature layers: {self.student_feature_layers}")
            teacher = YOLO(self.teacher_weights).model
        else:
            teacher = None

        return KDLoss(
            self,
            teacher_model=teacher,
            kd_type=self.kd_type,
            teacher_feature_layers=self.teacher_feature_layers,
            student_feature_layers=self.student_feature_layers,
            feature_beta=self.feature_beta,
        )


class KDTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        self.teacher_weights = overrides.pop("teacher", None)
        self.kd_type = overrides.pop("kd_type", "response")
        self.teacher_feature_layers = overrides.pop("teacher_feature_layers", None)
        self.student_feature_layers = overrides.pop("student_feature_layers", None)
        self.feature_beta = overrides.pop("feature_beta", 1.0)
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=False):
        model = KD_Model(
            cfg,
            teacher_weights=self.teacher_weights,
            kd_type=self.kd_type,
            teacher_feature_layers=self.teacher_feature_layers,
            student_feature_layers=self.student_feature_layers,
            feature_beta=self.feature_beta,
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

    kd_type = "feature"

    teacher_feature_layers = [15, 18, 21]  # YOLOv8 P3, P4, P5
    student_feature_layers = [16, 19, 22]  # YOLOv11 P3, P4, P5

    kd_alpha = 0.7
    kd_conf_thresh = 0.1
    kd_gamma = 1.0
    kd_temperature = 2.0
    feature_beta = 1.0  # Weight for feature loss

    dataset = "coco"
    epochs = 30
    imgsz = 640

    experiment_name = f"KD_{kd_type}_Y11n_from_Y8n_a{kd_alpha}_e{epochs}_{dataset}"

    experiment = Experiment(
        project_name="YOLO-Negative-flip",
    )

    experiment.set_name(experiment_name)

    hyperparams = {
        # KD hyperparameters
        "kd_type": kd_type,
        "kd_alpha": kd_alpha,
        "kd_conf_thresh": kd_conf_thresh,
        "kd_gamma": kd_gamma,
        "kd_temperature": kd_temperature,
        "feature_beta": feature_beta,
        "teacher_feature_layers": teacher_feature_layers
        if kd_type == "feature"
        else None,
        "student_feature_layers": student_feature_layers
        if kd_type == "feature"
        else None,
        # Training hyperparameters
        "epochs": epochs,
        "imgsz": imgsz,
        "dataset": dataset,
        # Model information
        "teacher_model": "yolov8n",
        "student_model": "yolo11n",
        # Loss configuration
        "loss_type": f"{kd_type}_kd",
        "distillation_type": "focal_positive_congruent"
        if kd_type == "response"
        else "l2_feature",
        "kd_components": "cls" if kd_type == "response" else "features",
    }
    experiment.log_parameters(hyperparams)

    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Hyperparameters: {hyperparams}")
    logger.info(f"Using {kd_type.upper()} distillation")

    trainer = KDTrainer(
        overrides={
            "model": student_model_path,
            "data": f"{dataset}.yaml",
            "epochs": epochs,
            "imgsz": imgsz,
            "device": "cuda",
            "teacher": teacher_model_path,
            "kd_type": kd_type,
            "teacher_feature_layers": teacher_feature_layers,
            "student_feature_layers": student_feature_layers,
            "feature_beta": feature_beta,
        }
    )

    trainer.train()
