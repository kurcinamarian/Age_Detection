import time
import matplotlib.pyplot as plt
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

CONFIG_DIR = "config"
ROOT_DIR = ""
CSV_PATH = os.path.join(ROOT_DIR, "train.csv")

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
    
    # Create confusion matrix images

    log_dict = {}

    unique_classes = sorted(set(y_true) | set(y_pred))
    n_classes = len(unique_classes)
    labels = list(range(3)) if class_names is None else list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Row-normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype(np.float32)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums!=0)

    figsize = (15,15)
    fig1, ax1 = plt.subplots(figsize=figsize)
    im = ax1.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax1.set_title(f"{title_prefix}Confusion Matrix (row-normalized)")
    ax1.set_xlabel('Predicted label')
    ax1.set_ylabel('True label')
    cbar = fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Proportion', rotation=90)

    ax1.set_xticks(range(n_classes))
    ax1.set_yticks(range(n_classes))
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

    model_name = cfg["model"]

    # Build model dynamically
    if model_name == "ResNet18":
        MODEL = models.resnet18(weights=None)
        MODEL.fc = nn.Linear(MODEL.fc.in_features, 3)
    elif model_name == "VGG-16":
        MODEL = models.vgg16(weights=None)
        MODEL.classifier[6] = nn.Linear(4096, 3)
    elif model_name == "EfficientNet-B0":
        MODEL = models.efficientnet_b0(weights=None)
        MODEL.classifier[1] = nn.Linear(MODEL.classifier[1].in_features, 3)
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
    
    # Make dataloaders
    train_loader, valid_loader, test_loader = create_dataloaders(
        csv_path=CSV_PATH,
        root_dir=ROOT_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
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
    
    # Loss function: CrossEntropyLoss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
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
    
    # Training loop
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, criterion, scaler)
        
        # Log detailed metrics every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(
                model, valid_loader, device, criterion, return_predictions=True)
            val_metrics = get_metrics(val_preds, val_labels, val_probs)
        else:
            val_loss, val_acc = evaluate(model, valid_loader, device, criterion, return_predictions=False)
            val_metrics = None
        
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
            # Log detailed metrics
            if val_metrics is not None:
                log_dict["val_f1_weighted"] = val_metrics["f1_weighted"]
                log_dict["val_precision_weighted"] = val_metrics["precision_weighted"]
                log_dict["val_recall_weighted"] = val_metrics["recall_weighted"]
                log_dict["val_f1_macro"] = val_metrics["f1_macro"]
                log_dict["val_precision_macro"] = val_metrics["precision_macro"]
                log_dict["val_recall_macro"] = val_metrics["recall_macro"]
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
        if epochs_no_improvement >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    if best_path.exists():
        # Load best checkpoint for test evaluation
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    # Final evaluation on all sets
    train_loss, train_acc, train_preds, train_labels, train_probs = evaluate(
        model, train_loader, device, criterion, return_predictions=True)
    
    val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(
        model, valid_loader, device, criterion, return_predictions=True)
    
    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, device, criterion, return_predictions=True)
    test_metrics = get_metrics(test_preds, test_labels, test_probs)
    
    print(f"TEST | loss={test_loss:.4f} acc={test_acc:.4f}")
    print(f"TEST | F1-weighted={test_metrics['f1_weighted']:.4f} "
          f"Precision={test_metrics['precision_weighted']:.4f} "
          f"Recall={test_metrics['recall_weighted']:.4f}")
    
    if USE_WANDB:
        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_acc,

            "val_loss": val_loss,
            "val_accuracy": val_acc,

            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_f1_weighted": test_metrics["f1_weighted"],
            "test_precision_weighted": test_metrics["precision_weighted"],
            "test_recall_weighted": test_metrics["recall_weighted"],
        })
        
        # Log confusion matrix
        cm_images = create_confusion_matrix_images(
            test_labels, test_preds,
            class_names=[str(i) for i in range(100)],
            title_prefix="final_test_"
        )
        wandb.log(cm_images)
    
        wandb.finish()


if __name__ == "__main__":
    configs = [f for f in os.listdir(CONFIG_DIR) if f.endswith(".json")]
    
    for cfg_file in configs:
        cfg_path = os.path.join(CONFIG_DIR, cfg_file)
        cfg = load_config(cfg_path)

        main(cfg)
