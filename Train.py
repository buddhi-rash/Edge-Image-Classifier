from NodeConvs_Net import NodeConvs_Net
from Data_loader import get_dataloaders
import torch
import torch.nn as nn
import numpy as np
import time
import copy
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def compute_eval_metrics(labels, probabilities):

    predictions = np.argmax(probabilities, axis=1)

    macro_auc = roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")

    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)

    macro_precision = precision_score(labels, predictions, average="macro", zero_division=0)

    macro_recall = recall_score(labels, predictions, average="macro", zero_division=0)

    accuracy = accuracy_score(labels, predictions)

    cm = confusion_matrix(labels, predictions)

    return {
        "auc": macro_auc,
        "f1": macro_f1,
        "precision": macro_precision,
        "recall": macro_recall,
        "accuracy": accuracy,
        "confusion_matrix": cm,
    }


def train_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = NodeConvs_Net(in_channels=3, base_channels= 32, levels= 3, dropout= 0.5, fc_depth= 1).to(device)
    print(f"Total trainable parameters: {model.count_trainable_parameters():,}")

    train_loader, val_loader, test_loader, (mean, std) = get_dataloaders(batch_size= 64, num_workers= 2)

    """counts = np.bincount(train_loader.dataset.labels, minlength=8 )
    weights = counts.sum() / (8 * counts)
    weights =  torch.tensor(weights, dtype=torch.float32, device=device)"""


    loss_critation = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-4, weight_decay= 1e-3)

    history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "val_auc": [], "val_f1": [],
            "epoch_time_sec": [], 
              # training pass only, not validation
            "lr": [],
        }

    best_val_auc = 0.0
    best_state = None
    EPOCHS = 20
    CHECKPOINT = "Best_modelA.pth"

    for epoch in range(EPOCHS):
        # Train the Model
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        train_start = time.time()
        with torch.enable_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                predicts = model(images)

                loss = loss_critation(predicts, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                predictions = predicts.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_train_loss = total_loss / total
        train_accuracy = correct / total

        train_time = time.time() - train_start

        #Validation Step
        model.eval()
        #model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        val_probs_list = []
        val_targets_list = []

        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                predicts = model(images)
                val_loss = loss_critation(predicts, labels)

                total_loss += val_loss.item() * images.size(0)
                total += labels.size(0)

                probs = F.softmax(predicts, dim=1)
                val_probs_list.append(probs.cpu().numpy())
                val_targets_list.append(labels.cpu().numpy())

        avg_val_loss = total_loss / total
        val_probs = np.concatenate(val_probs_list, axis=0)
        val_targets = np.concatenate(val_targets_list, axis=0)


        val_metrics = compute_eval_metrics(val_targets, val_probs)
        current_lr = optimizer.param_groups[0]["lr"]
 
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_accuracy)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_f1"].append(val_metrics["f1"])
        history["epoch_time_sec"].append(train_time)
        history["lr"].append(current_lr)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, CHECKPOINT)

        print(
            f"Epoch {epoch+1:2d}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | train_Acc: {train_accuracy:.4f} || "
            f"Val Loss: {avg_val_loss:.4f} | Val_Acc: {val_metrics['accuracy']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f} | F1: {val_metrics['f1']:.4f} | "
            f"Time: {train_time:.1f}s"
        )


    avg_epoch_time = sum(history["epoch_time_sec"]) / len( history["epoch_time_sec"])
    print(f"\nBest Validation Macro AUC: {best_val_auc:.4f}")
    print(f"Average Training Time / Epoch: {avg_epoch_time:.1f}s")
    


if __name__ == "__main__":
    train_model()