import time
from collections import Counter
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from dataloader import create_dataloaders
from torchvision import models

CONFIG_DIR = "configs"
ROOT_DIR = ""

@torch.no_grad()
def evaluate(model, loader, device, criterion, return_predictions=False):

    model.eval()

    total_loss, total, correct = 0.0, 0, 0
    all_preds = []      # Predicted class_ids
    all_labels = []     # True labels
    all_probs = []      # Predicted probability for each class

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        if return_predictions:
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)

    if return_predictions:
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)

    return avg_loss, accuracy

def get_metrics(y_true, y_pred, y_probs):

    # Calculte metrics for evaluation

    num_classes = 3

    # Weighted average metrics (f1, precision, recall)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate per-class specificity
    specificity_per_class = []
    for i in range(num_classes):
        true_negative = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        false_positive = cm[:, i].sum() - cm[i, i]
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive > 0) else 0
        specificity_per_class.append(specificity)

    metrics = {
        'f1_weighted': f1_weighted,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'confusion_matrix': cm,
        'classification_report': report,
        'specificity_per_class': specificity_per_class
    }

    return metrics


def create_confusion_matrix_images(y_true, y_pred, class_names=None, title_prefix=""):
    """
    Creates a confusion matrix image with raw counts and numbers inside each cell.
    Returns a dictionary with a wandb.Image.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import wandb

    log_dict = {}

    unique_classes = sorted(set(y_true) | set(y_pred))
    n_classes = len(unique_classes)
    labels = list(range(n_classes)) if class_names is None else list(range(len(class_names)))

    # Raw confusion matrix (counts)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    figsize = (10, 10)  # smaller size for raw counts
    fig1, ax1 = plt.subplots(figsize=figsize)
    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_title(f"{title_prefix}Confusion Matrix")
    ax1.set_xlabel('Predicted label')
    ax1.set_ylabel('True label')
    cbar = fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Count', rotation=90)

    # Add numbers inside each cell
    for i in range(n_classes):
        for j in range(n_classes):
            value = cm[i, j]
            ax1.text(
                j, i,
                f"{value}",
                ha="center", va="center",
                color="black" if cm[i, j] < cm.max() / 2 else "white",
                fontsize=10
            )

    ax1.set_xticks(range(n_classes))
    ax1.set_yticks(range(n_classes))
    if class_names is not None:
        ax1.set_xticklabels(class_names, rotation=90, fontsize=8)
        ax1.set_yticklabels(class_names, fontsize=8)

    plt.tight_layout()
    log_dict[f"{title_prefix}confusion_matrix_img"] = wandb.Image(fig1)
    plt.close(fig1)

    return log_dict


def train_one_epoch(model, loader, device, optimizer, criterion, scaler):

    # Training for one epoch, return average loss and accuracy

    model.train()
    total_loss, total, correct = 0.0, 0, 0
    for images, labels in loader:
        # Get images and labels
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True) # Clear gradients

        if device.type == 'cuda': # GPU training with scaler
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: # CPU training
            with torch.amp.autocast(device_type="cpu"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)


    return (total_loss / max(1, total)), (correct / max(1, total))


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc):
    # Save the current state
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_acc": best_acc,
    }, path)

def load_config(json_path):
    with open(json_path, "r") as f:
        cfg = json.load(f)
    return cfg

def main(cfg):

    DATASET_PATH = cfg["dataset_path"]

    model_name = cfg["model"]

    # Build model dynamically
    if model_name == "ResNet18":
        MODEL = models.resnet18(weights=None)
        for param in MODEL.layer1.parameters():
            param.requires_grad = False
        MODEL.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(MODEL.fc.in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3)
        )
    elif model_name == "ResNet18-2":
        MODEL = models.resnet18(weights=None)
        for param in MODEL.layer1.parameters():
            param.requires_grad = False
        MODEL.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(MODEL.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )
    elif model_name == "ResNet18-3":
        MODEL = models.resnet18(weights=None)
        for param in MODEL.layer1.parameters():
            param.requires_grad = False
        MODEL.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(MODEL.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)
        )
    elif model_name == "VGG-16":
        MODEL = models.vgg16(weights=None)
        MODEL.classifier[6] = nn.Linear(4096, 3)
    elif model_name == "EfficientNet-B0":
        MODEL = models.efficientnet_b0(weights=None)
        for param in MODEL.features[0].parameters():
            param.requires_grad = False
        MODEL.classifier[1] = nn.Linear(MODEL.classifier[1].in_features, 3)
    elif model_name == "MobileNetV3-Large":
        MODEL = models.mobilenet_v3_large(weights=None)
        MODEL.classifier = nn.Sequential(
            nn.Linear(MODEL.classifier[0].in_features, 128),
            nn.Hardswish(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    EPOCHS = cfg["epochs"]
    BATCH_SIZE = cfg["batch_size"]
    LR = cfg["lr"]
    WEIGHT_DECAY = cfg["weight_decay"]
    LABEL_SMOOTHING = cfg["label_smoothing"]
    NUM_WORKERS = cfg["num_workers"]
    EARLY_STOP_PATIENCE = cfg["early_stop_patience"]
    OUT_DIR = Path(cfg["out_dir"])
    USE_WANDB = cfg["use_wandb"]
    WANDB_PROJECT = cfg["wandb_project"]
    WANDB_ENTITY = cfg["wandb_entity"]
    WANDB_RUN_NAME = cfg["wandb_run_name"]
    WANDB_LOG_FREQUENCY = cfg["wandb_log_frequency"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    # Make dataloaders
    train_loader, valid_loader, test_loader = create_dataloaders(
        BATCH_SIZE=BATCH_SIZE,
        DATASET_PATH=DATASET_PATH,
        NUM_WORKERS=NUM_WORKERS
    )

    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=WANDB_RUN_NAME,
            config={
                "model": model_name,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "label_smoothing": LABEL_SMOOTHING,
                "early_stopping_patience": EARLY_STOP_PATIENCE,
            }
        )

    model = MODEL.to(device)

    if USE_WANDB:
        wandb.watch(model, log="all", log_freq=WANDB_LOG_FREQUENCY)
    all_labels = []

    for _, label in train_loader:  # train_loader is your DataLoader
        all_labels.extend(label.tolist())  # gather all labels

    class_counts = Counter(all_labels)
    n_classes = len(class_counts)
    total_samples = sum(class_counts.values())
    class_weights = []

    for i in range(n_classes):
        class_weight = total_samples / (n_classes * class_counts[i])
        class_weights.append(class_weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = class_weights.to(device)
    # Loss function: CrossEntropyLoss with label smoothing and class weights
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING,weight=class_weights)

    # Optimizer: Adam with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler: CosineAnnealingLR (gradual LR decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Scaler: Mixed precision training for faster GPU training -> usage suggested by ChatGPT
    scaler = torch.amp.GradScaler(device.type if device.type == "cuda" else "cpu")

    best_path = OUT_DIR / "best.pt"
    last_path = OUT_DIR / "last.pt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0 # current best validation accuracy
    epochs_no_improvement = 0 # Count epoch without improvement for early stopping

    print("started training")

    # Training loop
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion, scaler)

        val_loss, val_acc = evaluate(model, valid_loader, device, criterion, return_predictions=False)

        scheduler.step()
        dt = time.time() - t0

        print(
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
            f"train {train_loss:.4f}/{train_acc:.4f} | "
            f"val {val_loss:.4f}/{val_acc:.4f} | "
            f"{dt:.1f}s")

        if USE_WANDB:
            # Log metrics
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "lr": optimizer.param_groups[0]["lr"]
            }

            wandb.log(log_dict)

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val_acc) # Save current state

        # Update best state (Save new best state if it is better than the current best state)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improvement = 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val_acc)
            print(f"New best acc: {best_val_acc:.4f} -> {best_path}")
        else:
            epochs_no_improvement += 1
        # If too many epochs without improvement, stop early
        if train_acc - val_acc > 0.3 and val_acc > 88: #epochs_no_improvement >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_path.exists():
        # Load best checkpoint for test evaluation
        checkpoint = torch.load(best_path, map_location=device,  weights_only=True)
        model.load_state_dict(checkpoint["model_state"])

    # Final evaluation on all sets
    train_loss, train_acc, train_preds, train_labels, train_probs = evaluate(
        model, train_loader, device, criterion, return_predictions=True)

    val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(
        model, valid_loader, device, criterion, return_predictions=True)

    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, device, criterion, return_predictions=True)
    test_metrics = get_metrics(test_labels, test_preds, test_probs)

    print(f"TEST | loss={test_loss:.4f} acc={test_acc:.4f}")
    print(f"TEST | F1-weighted={test_metrics['f1_weighted']:.4f} "
          f"Precision={test_metrics['precision_weighted']:.4f} "
          f"Recall={test_metrics['recall_weighted']:.4f}")

    if USE_WANDB:
        # Log metrics
        wandb.log({
            "best/train_loss": train_loss,
            "best/train_accuracy": train_acc,

            "best/val_loss": val_loss,
            "best/val_accuracy": val_acc,

            "best/test_loss": test_loss,
            "best/test_accuracy": test_acc,
            "best/test_f1_weighted": test_metrics["f1_weighted"],
            "best/test_precision_weighted": test_metrics["precision_weighted"],
            "best/test_recall_weighted": test_metrics["recall_weighted"],
        })

        # Log confusion matrix
        cm_images = create_confusion_matrix_images(
            test_labels, test_preds,
            class_names=["MIDDLE","OLD","YOUNG"],
            title_prefix="final_test_"
        )
        wandb.log(cm_images)

        wandb.finish()


if __name__ == "__main__":
    configs = [f for f in os.listdir(CONFIG_DIR) if f.endswith(".json")]

    for cfg_file in configs:
        print("Starting training with:", cfg_file)
        cfg_path = os.path.join(CONFIG_DIR, cfg_file)
        cfg = load_config(cfg_path)

        main(cfg)
