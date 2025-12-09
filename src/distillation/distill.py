import comet_ml
import torch
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from comet_ml import Experiment


# --- Custom Loss Function ---
class KDLoss(v8DetectionLoss):
    def __init__(self, model, teacher_model=None, alpha=0.5, T=2.0):
        super().__init__(model)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.T = T
        self.nc = model.nc
        self.reg_max = getattr(model.model[-1], "reg_max", 16)

        if self.teacher_model:
            self.teacher_model.eval()
            device = next(model.parameters()).device
            self.teacher_model.to(device)
            #  Freeze teacher
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def __call__(self, preds, batch):
        # 1. Standard Loss (Handles tuple extraction internally)
        loss, loss_items = super().__call__(preds, batch)

        if self.teacher_model is None:
            return loss, loss_items

        # During validation, preds is (decoded_out, [feat1, feat2, feat3])
        # We need the list of features, which is index 1
        student_feats = preds[1] if isinstance(preds, tuple) else preds

        # 2. Teacher Forward Pass
        with torch.no_grad():
            # Handle FP16/FP32 Mismatch
            input_tensor = batch["img"]
            teacher_dtype = next(self.teacher_model.parameters()).dtype

            if input_tensor.dtype != teacher_dtype:
                input_tensor = input_tensor.to(teacher_dtype)

            teacher_outputs = self.teacher_model(input_tensor)

            # Extract raw logits (index 1 if tuple)
            if isinstance(teacher_outputs, tuple):
                teacher_preds = teacher_outputs[1]
            else:
                teacher_preds = teacher_outputs

        # 3. Compute Distillation Loss using extracted features
        dist_loss = self.compute_distill_loss(student_feats, teacher_preds)

        # 4. Combine & Log
        total_loss = loss + (self.alpha * dist_loss)

        dist_loss_val = dist_loss.detach()
        loss_items = torch.cat([loss_items, dist_loss_val.view(1)])

        return total_loss, loss_items

    def compute_distill_loss(self, student_preds, teacher_preds):
        total_dist_loss = 0.0

        for s_pred, t_pred in zip(student_preds, teacher_preds):
            # Shape matching
            if s_pred.shape[-2:] != t_pred.shape[-2:]:
                t_pred = F.interpolate(t_pred, size=s_pred.shape[-2:], mode="nearest")

            # Ensure Teacher preds match Student preds dtype
            if t_pred.dtype != s_pred.dtype:
                t_pred = t_pred.to(s_pred.dtype)

            split_idx = 4 * self.reg_max

            # Class Loss
            s_cls = s_pred[:, split_idx:, :, :]
            t_cls = t_pred[:, split_idx:, :, :]

            cls_loss = F.kl_div(
                F.log_softmax(s_cls / self.T, dim=1),
                F.softmax(t_cls / self.T, dim=1),
                reduction="batchmean",
            ) * (self.T**2)

            # BBox Loss
            s_box = s_pred[:, :split_idx, :, :]
            t_box = t_pred[:, :split_idx, :, :]

            box_loss = F.kl_div(
                F.log_softmax(s_box / self.T, dim=1),
                F.softmax(t_box / self.T, dim=1),
                reduction="batchmean",
            ) * (self.T**2)

            total_dist_loss += cls_loss + box_loss

        return total_dist_loss / len(student_preds)


# --- 2. Custom Model Wrapper ---
class KD_Model(DetectionModel):
    def __init__(self, cfg, teacher_weights, **kwargs):
        super().__init__(cfg, **kwargs)
        self.teacher_weights = teacher_weights

    def init_criterion(self):
        """Initialize custom loss logic with teacher"""
        # Load teacher model
        if self.teacher_weights:
            print(f"Loading teacher model for distillation: {self.teacher_weights}")
            teacher = YOLO(self.teacher_weights).model
            teacher.eval()
        else:
            teacher = None

        return KDLoss(self, teacher_model=teacher)


# --- 3. Custom Trainer ---
class KDTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}

        # Extract custom argument and remove it from overrides
        self.teacher_weights = overrides.pop("teacher", None)

        # Initialize parent
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        # Create Custom Model passing the stored teacher weights
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
        # Add 'dist_loss' to logging
        keys = ["box_loss", "cls_loss", "dfl_loss", "dist_loss"]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip([f"{prefix}/{x}" for x in keys], loss_items))
        else:
            return keys


# --- 4. Execution ---
if __name__ == "__main__":
    teacher_model_path = "yolov8n.pt"
    student_model_path = "yolo11n.pt"

    experiment = Experiment(
        project_name="YOLO-Negative-flip",
    )

    experiment.set_name("KD training")

    trainer = KDTrainer(
        overrides={
            "model": student_model_path,
            "data": "coco.yaml",
            "epochs": 10,
            "imgsz": 640,
            "device": "cuda",
            "teacher": teacher_model_path,
        }
    )

    trainer.train()
