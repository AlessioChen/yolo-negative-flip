from comet_ml import Experiment

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD, lr_scheduler

import os
import random 
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

from COCODataset import COCODataset
from detector import Detector



load_dotenv()

def collate_fn(batch):
    """
    Custom collate function to handle batches of images and targets
    batch: list of (image, target) tuples
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets


def train_one_epoch(model, optimizer, data_loader, device, experiment, epoch, cls_loss_fn, bbox_loss_fn):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_bbox_loss = 0

    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc=f"Batch {batch_idx + 1}")):
        batch_loss = 0
        batch_cls_loss = 0
        batch_bbox_loss = 0
        num_objects = 0
        
        for i in range(len(images)):
            image = images[i].to(device)  
            target = {k: v.to(device) for k, v in targets[i].items()} 

            for j in range(len(target['labels'])):
                target_label = target['labels'][j]
                target_box = target['boxes'][j]
                
                cls_score, bbox_pred = model(image)
                
                cls_loss = cls_loss_fn(cls_score.unsqueeze(0), target_label.unsqueeze(0))
                bbox_loss = bbox_loss_fn(bbox_pred.unsqueeze(0), target_box.unsqueeze(0))
                
                batch_cls_loss += cls_loss
                batch_bbox_loss += bbox_loss
                num_objects += 1
                
        
        if num_objects > 0:
            batch_cls_loss = batch_cls_loss / len(images)
            batch_bbox_loss = batch_bbox_loss / len(images)
            batch_loss = batch_cls_loss + batch_bbox_loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            total_cls_loss += batch_cls_loss.item()
            total_bbox_loss += batch_bbox_loss.item()
            
            # Log batch metrics
            if experiment:
                global_step = epoch * len(data_loader) + batch_idx
                experiment.log_metrics({
                    'batch_loss': batch_loss.item(),
                    'batch_cls_loss': batch_cls_loss.item(),
                    'batch_bbox_loss': batch_bbox_loss.item()
                }, step=global_step)

    return total_loss / len(data_loader), total_cls_loss / len(data_loader), total_bbox_loss / len(data_loader)


def init_args(): 
    parser = argparse.ArgumentParser(description='Train detector model')
    parser.add_argument('--use_half', action='store_true', help='Use half dataset for training')
    parser.add_argument('--size', type=int, default=50000, help='Size of dataset to use')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='detector.pth', help='Model save path')
    parser.add_argument('--use_cometml', action='store_true', help='Enable Comet ML experiment logging')
    args = parser.parse_args()
    
    return args
    

def main(): 
    args = init_args()
    
    torch.manual_seed(22)
    random.seed(22)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    print(f"Using device: {device}")
    
    # Initialize Comet ML experiment
    experiment = None 
    if args.use_cometml:
        experiment = Experiment(
            api_key=os.getenv('COMET_API_KEY'),
            project_name="negative-flips-detector"
        )
    
        experiment.log_parameter('args', args)

    dataset = COCODataset(
        root_dir='../data/train2017', 
        ann_file='../data/annotations/instances_train2017.json',
        image_size=(224, 224)
    )
    
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    dataset_size = args.size // 2 if args.use_half else args.size
    selected_indices = indices[:dataset_size]
    dataset = Subset(dataset, selected_indices)
    print(f"Training on {len(dataset)} images")
    
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    model = Detector().to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    cls_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()
    
    best_loss = float('inf')
    
    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        train_loss, cls_loss, bbox_loss = train_one_epoch(
            model, optimizer, data_loader, device, experiment, 
            epoch, cls_loss_fn, bbox_loss_fn
        )
        
        print(f"[Epoch {epoch+1}] Total Loss: {train_loss:.4f}, Cls Loss: {cls_loss:.4f}, BBox Loss: {bbox_loss:.4f}")

        
        # Log epoch metrics
        if experiment:
            experiment.log_metrics({
                'epoch_loss': train_loss,
                'epoch_cls_loss': cls_loss,
                'epoch_bbox_loss': bbox_loss,
                'lr': optimizer.param_groups[0]['lr']
            }, step=epoch)
        
        scheduler.step()
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), args.save_path)
            if experiment:
                experiment.log_model('best_detector', args.save_path)

    
if __name__ == '__main__':
    main()