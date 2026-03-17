#main
import torch
import os
import sys
import random
import subprocess

# Ensure ultralytics is installed for YOLOv5
try:
    import ultralytics
except ImportError:
    print("ultralytics not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    import ultralytics

# Ensure gitpython is installed
try:
    import git
except ImportError:
    print("gitpython not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gitpython"])
    print("gitpython installed.")

# Local imports
import config
from Utils import OxfordPetDataset, PennFudanDataset, get_transform, collate_fn, create_dataloader, yolov5_collate_fn, letterbox
from rcnn import get_faster_rcnn_model
from YOLO import get_yolov5_model
from train import train_model
from evalu import evaluate_model


# FIX #3A — Helper class so all three splits share the same dataset instance,
# avoiding non-deterministic os.listdir() ordering across separate instantiations.
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.subset[idx]
        if self.transform:
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        return len(self.subset)


def run_experiment(model_type, dataset_name, num_classes, dataset_root, classes_list):
    print(f"\n{'='*50}")
    print(f"Starting experiment: Model = {model_type}, Dataset = {dataset_name}")
    print(f"{'='*50}\n")

    # 1. Data Loading and Splitting
    print("Loading and splitting dataset...")

    if dataset_name == 'Oxford-IIIT Pet':
        # FIX #3B — Create dataset once with transforms=None, apply transforms per-split
        # via TransformSubset so all indices refer to the same file ordering.
        full_dataset = OxfordPetDataset(root=dataset_root, classes=classes_list, transforms=None)
        dataset_size = len(full_dataset)
        indices = list(range(dataset_size))
        random.shuffle(indices)

        train_split = int(0.7 * dataset_size)
        val_split   = int(0.15 * dataset_size)
        train_indices = indices[:train_split]
        val_indices   = indices[train_split:train_split + val_split]
        test_indices  = indices[train_split + val_split:]

        train_dataset = TransformSubset(torch.utils.data.Subset(full_dataset, train_indices), get_transform(train=True))
        val_dataset   = TransformSubset(torch.utils.data.Subset(full_dataset, val_indices),   get_transform(train=False))
        test_dataset  = TransformSubset(torch.utils.data.Subset(full_dataset, test_indices),  get_transform(train=False))

    elif dataset_name == 'Penn-Fudan Pedestrian':
        # FIX #3C — Same pattern applied to Penn-Fudan
        full_dataset = PennFudanDataset(root=dataset_root, transforms=None)
        dataset_size = len(full_dataset)
        indices = list(range(dataset_size))
        random.shuffle(indices)

        train_split = int(0.7 * dataset_size)
        val_split   = int(0.15 * dataset_size)
        train_indices = indices[:train_split]
        val_indices   = indices[train_split:train_split + val_split]
        test_indices  = indices[train_split + val_split:]

        train_dataset = TransformSubset(torch.utils.data.Subset(full_dataset, train_indices), get_transform(train=True))
        val_dataset   = TransformSubset(torch.utils.data.Subset(full_dataset, val_indices),   get_transform(train=False))
        test_dataset  = TransformSubset(torch.utils.data.Subset(full_dataset, test_indices),  get_transform(train=False))

        print(f"Train/Val/Test sizes: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")
        from collections import Counter
        label_counts = Counter()
        for _, target in train_dataset:
            label_counts.update(target['labels'].tolist())
        print(f"Label distribution: {dict(sorted(label_counts.items()))}")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 2. Create DataLoaders
    print("Creating DataLoaders...")
    if model_type == config.YOLOV5N_MODEL_NAME:
        train_dataloader = create_dataloader(train_dataset, config.BATCH_SIZE, shuffle=True,  num_workers=0, collate_fn=yolov5_collate_fn)
        val_dataloader   = create_dataloader(val_dataset,   config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=yolov5_collate_fn)
        test_dataloader  = create_dataloader(test_dataset,  config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=yolov5_collate_fn)
    else:
        train_dataloader = create_dataloader(train_dataset, config.BATCH_SIZE, shuffle=True,  num_workers=0, collate_fn=collate_fn)
        val_dataloader   = create_dataloader(val_dataset,   config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
        test_dataloader  = create_dataloader(test_dataset,  config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # 3. Model Initialization
    print("Initializing model...")
    if model_type == config.FASTER_RCNN_MODEL_NAME:
        model = get_faster_rcnn_model(num_classes=num_classes)
    elif model_type == config.YOLOV5N_MODEL_NAME:
        model = get_yolov5_model(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.to(config.DEVICE)

    # 4. Optimizer and LR Scheduler
    print("Setting up optimizer and learning rate scheduler...")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 5. Training
    print("Starting training...")
    total_train_time = train_model(
        model,
        model_type,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        config.NUM_EPOCHS,
        config.DEVICE,
        config.CHECKPOINT_DIR
    )

    # 6. Load best model for evaluation
    print("Loading best model for evaluation...")
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{model_type}_best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded successfully.")
        except RuntimeError as e:
            print(f"Checkpoint mismatch, skipping load: {e}")
        print(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {checkpoint['best_val_loss']:.4f}")
    else:
        print(f"Warning: No best model checkpoint found at {checkpoint_path}. Evaluating the last trained model state.")

    # 7. Evaluation
    print("Starting evaluation...")
    eval_metrics = evaluate_model(
        model,
        test_dataloader,
        config.DEVICE,
        model_type,
        num_classes=num_classes
    )

    print(f"\nResults for {model_type} on {dataset_name}:")
    print(f"  Total Training Time: {total_train_time:.2f} seconds")
    print(f"  mAP@0.5: {eval_metrics['mAP@0.5']:.4f}%")
    print(f"  Precision (COCO mAP): {eval_metrics['precision']:.4f}%")
    print(f"  Recall (MAR@100): {eval_metrics['recall']:.4f}%")
    print(f"  Average Inference Speed: {eval_metrics['avg_inference_speed_per_image_sec']:.4f} seconds/image")

    return total_train_time, eval_metrics


if __name__ == "__main__":
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    all_results = {}

    train_time, metrics = run_experiment(
        model_type=config.ACTIVE_MODEL,
        dataset_name=config.ACTIVE_DATASET,
        num_classes=config.ACTIVE_NUM_CLASSES,
        classes_list=config.ACTIVE_CLASSES_LIST,
        dataset_root=config.ACTIVE_DATASET_ROOT
    )
    all_results[f'{config.ACTIVE_MODEL}_{config.ACTIVE_DATASET}'] = {'train_time': train_time, **metrics}

    print("\n\n{'#'*50}")
    print("\nOverall Experiment Summary\n")
    print("{'#'*50}\n")

    for combo, results in all_results.items():
        print(f"-- {combo} --")
        print(f"  Total Training Time: {results['train_time']:.2f} seconds")
        print(f"  mAP@0.5: {results['mAP@0.5']:.4f}%")
        print(f"  Precision (COCO mAP): {results['precision']:.4f}%")
        print(f"  Recall (MAR@100): {results['recall']:.4f}%")
        print(f"  Average Inference Speed: {results['avg_inference_speed_per_image_sec']:.4f} seconds/image")
        print("\n")

    print("Selected experiment completed!")