import torch 
import torchvision
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os 
from PIL import Image 
from typing import Optional, Callable, List, Tuple, Dict


class COCODataset(Dataset):
    def __init__(self, root_dir: str, ann_file: str, transform: Optional[Callable] = None, image_size: Tuple[int, int] = (224, 224)) -> None:
        self.root_dir: str = root_dir
        self.coco: COCO = COCO(ann_file)
        self.ids: List[int] = list(sorted(self.coco.imgs.keys()))
        self.transform: Optional[Callable] = transform
        self.image_size = image_size
        
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict]:
        img_id = self.ids[idx]

        # Load Image 
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(f"{self.root_dir}/{img_info['file_name']}")
        image = Image.open(img_path).convert('RGB')

        # Load Annotations 
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns: 
            if ann['iscrowd'] == 0:  # skip crowd annotations 
                x,y,w,h = ann['bbox'] 
                # Convert to [x1, y1, x2, y2]
                bbox = [x, y, x + w, y + h]
                boxes.append(bbox)
                labels.append(ann['category_id'] - 1) # Convert to 0-indexed

        # Convert to tensors 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }

        if self.transform:
            image = self.transform(image)
        else: 
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.image_size)),
                torchvision.transforms.ToTensor()
            ])
            image = transform(image)

        return image, target
        