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
    images = torch.stack([item[0] for item in batch], dim=0)  # [B, 3, H, W]
    targets = [item[1] for item in batch]  
    return images, targets


def prepare_targets_batch(images, targets):

    filtered_data = [(img, t) for img, t in zip(images, targets) if len(t['labels']) > 0 and len(t['boxes']) > 0]
    if len(filtered_data) == 0:
        return images, None, None  
    
    filtered_images, filtered_targets = zip(*filtered_data)
    
    labels = torch.stack([t['labels'][0] for t in filtered_targets])
    boxes = torch.stack([t['boxes'][0] for t in filtered_targets])
    
    images = torch.stack(filtered_images)
    
    return images, labels, boxes


def train_one_epoch(model, optimizer, data_loader, device, experiment, epoch, cls_loss_fn, bbox_loss_fn):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_batches = 0

    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc=f"Training Epoch {epoch+1}")):
        images, labels, boxes = prepare_targets_batch(images, targets)
        if images is None:
            continue

        images = images.to(device)  
        labels = labels.to(device)
        boxes = boxes.to(device)

        cls_scores, bbox_preds = model(images)  
        cls_loss = cls_loss_fn(cls_scores, labels)
        bbox_loss = bbox_loss_fn(bbox_preds, boxes)
        loss = cls_loss + bbox_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_loss = loss.item()
        batch_cls_loss = cls_loss.item()
        batch_bbox_loss = bbox_loss.item()

        total_loss += batch_loss
        total_cls_loss += batch_cls_loss
        total_bbox_loss += batch_bbox_loss
        total_batches += 1

        if experiment:
            global_step = epoch * len(data_loader) + batch_idx
            experiment.log_metrics({
                'batch_loss': batch_loss,
                'batch_cls_loss': batch_cls_loss,
                'batch_bbox_loss': batch_bbox_loss
            }, step=global_step)

    if total_batches == 0:
        return 0, 0, 0
    return total_loss / total_batches, total_cls_loss / total_batches, total_bbox_loss / total_batches


def validate_one_epoch(model, data_loader, device, cls_loss_fn, bbox_loss_fn):
    model.eval()
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validation"):
            
            images, labels, boxes = prepare_targets_batch(images, targets)
            if labels is None or boxes is None:
                continue
            
            images = images.to(device)
            labels = labels.to(device)
            boxes = boxes.to(device)

            cls_scores, bbox_preds = model(images)
            cls_loss = cls_loss_fn(cls_scores, labels)
            bbox_loss = bbox_loss_fn(bbox_preds, boxes)
            loss = cls_loss + bbox_loss

            total_cls_loss += cls_loss.item()
            total_bbox_loss += bbox_loss.item()
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches, total_cls_loss / num_batches, total_bbox_loss / num_batches


def init_args(): 
    parser = argparse.ArgumentParser(description='Train detector model')
    parser.add_argument('--use_half', action='store_true', help='Use half dataset for training')
    parser.add_argument('--size', type=int, default=100000, help='Size of dataset to use')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
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
        experiment.log_parameters(vars(args))

    dataset = COCODataset(
        root_dir='../data/train2017', 
        ann_file='../data/annotations/instances_train2017.json',
        image_size=(224, 224)
    )
    

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    dataset_size = args.size // 2 if args.use_half else args.size
    selected_indices = indices[:dataset_size]

    val_size = int(0.1 * len(selected_indices))
    val_indices = selected_indices[:val_size]
    train_indices = selected_indices[val_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Training on {len(train_dataset)} images")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,  # usa batch anche in validation
        shuffle=False,
        collate_fn=collate_fn
    )

    model = Detector().to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    cls_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()
        
    for epoch in tqdm(range(args.epochs), desc="Training Epochs"):
        train_loss, cls_loss, bbox_loss = train_one_epoch(
            model, optimizer, train_loader, device, experiment, 
            epoch, cls_loss_fn, bbox_loss_fn
        )
        
        val_loss, val_cls_loss, val_bbox_loss = validate_one_epoch(
            model, val_loader, device, cls_loss_fn, bbox_loss_fn
        )

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"[Cls: {cls_loss:.4f}, BBox: {bbox_loss:.4f}]")
        print(f"[Val Cls: {val_cls_loss:.4f}, Val BBox: {val_bbox_loss:.4f}]")

        if experiment:
            experiment.log_metrics({
                'epoch_loss': train_loss,
                'epoch_cls_loss': cls_loss,
                'epoch_bbox_loss': bbox_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'val_loss': val_loss,
                'val_cls_loss': val_cls_loss,
                'val_bbox_loss': val_bbox_loss
            }, step=epoch)

        scheduler.step()
            
    torch.save(model.state_dict(), args.save_path)        
    print(f"Training complete!")
    print(f"Model saved to {args.save_path}")
    
    if experiment:
        experiment.end()
    
if __name__ == '__main__':
    main()
