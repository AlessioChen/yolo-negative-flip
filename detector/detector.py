import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Detector(nn.Module):
    def __init__(self, num_classes=80):
        super(Detector, self).__init__()
        
        # Backbone (simple CNN)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.fc_cls = nn.Linear(128, num_classes)
        self.fc_bbox = nn.Linear(128, 4)

    def forward(self, x):
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        features = self.backbone(x) # (1, 128, 1, 1)
        features = features.view(features.size(0), -1) # [1, 128]

        cls_scores = self.fc_cls(features) # [1, num_classes]
        bbox_preds = self.fc_bbox(features) # [1, 4]

        return cls_scores.squeeze(0), bbox_preds.squeeze(0) 