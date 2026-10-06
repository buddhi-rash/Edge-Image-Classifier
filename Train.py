from NodeConvs_A_Net import NodeConvs_Net
from NodeConvs_B_Net import NodeConvs_B_Net
from Data_loader import get_dataloaders
import torch
import torch.nn as nn
import numpy as np
import time
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
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


def train_model(model, learning_rate,  epochs, optimizer_type, Save_path, train_loader, val_loader, state_dict_path=None, weight_decay=1e-4, momentum=0.9, history=None, best_val_f1=0.0, device=None):
      
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    print(f"Training Model {model.__class__.__name__} with {model.count_trainable_parameters():,} trainable parameters.")

    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path, weights_only=True)
        model.load_state_dict(state_dict)
        print("loaded the best saved model")

    

    """counts = np.bincount(train_loader.dataset.labels, minlength=8 )
    weights = counts.sum() / (8 * counts)
    weights =  torch.tensor(weights, dtype=torch.float32, device=device)"""


    loss_critation = nn.CrossEntropyLoss()

    if optimizer_type == "adam":
        optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0, weight_decay=weight_decay)
    elif optimizer_type == "sgd_momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    if history is None:
        history = {
                "train_loss": [], "val_loss": [],
                "train_acc": [], "val_acc": [],
                "val_auc": [], "val_f1": [],
                "epoch_time_sec": [], 
                "lr": [],
            }

    
    best_state = None
    EPOCHS = epochs
    CHECKPOINT = Save_path

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

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
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
    print(f"\nBest Validation Macro f1: {best_val_f1:.4f}")
    print(f"Average Training Time / Epoch: {avg_epoch_time:.1f}s")

    return model, history, best_state, best_val_f1
    
 
def plot_loss_curves(history, model, optimizer_type, save_path, phase_boundaries=None):
    """phase_boundaries: epoch numbers (1-indexed) where a new phase
    started (e.g. where the LR changed), drawn as dashed vertical lines."""

    title = model.__class__.__name__
    epochs = range(1, len(history["train_loss"]) + 1)
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title}: Loss trained with {optimizer_type}")
    axes[0].legend()
 
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title}: Accuracy trained with {optimizer_type} ")
    axes[1].legend()
 
    if phase_boundaries:
        for ax in axes:
            for boundary in phase_boundaries:
                ax.axvline(boundary + 0.5, color="gray", linestyle="--", alpha=0.6)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved {save_path}")

def model_comparison(model_A, model_B, optimizer_type, Save_path_A, Save_path_B, train_loader, val_loader, batch_size=64, num_workers=2, weight_decay=1e-4, momentum=0.9):
    print("Training Model A...")
    model_A_trained, history_A, best_state_A, best_val_f1_A = train_model(
        model=model_A,
        learning_rate=1e-3,
        epochs=20,
        optimizer_type=optimizer_type,
        Save_path=Save_path_A,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        weight_decay=weight_decay,
        momentum=momentum
    )

    model_A_trained, history_A, best_state_A, best_val_f1_A = train_model(
        model=model_A,
        learning_rate=1e-5,
        epochs=30,
        optimizer_type=optimizer_type,
        state_dict_path = Save_path_A,
        Save_path=Save_path_A,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        weight_decay=weight_decay,
        momentum=momentum,
        history=history_A,
        best_val_f1=best_val_f1_A
    )




    print("\nTraining Model B...")
    model_B_trained, history_B, best_state_B, best_val_f1_B = train_model(
        model=model_B,
        learning_rate=1e-3,
        epochs=20,
        optimizer_type=optimizer_type,
        Save_path=Save_path_B,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        weight_decay=weight_decay,
        momentum=momentum
    )

    model_B_trained, history_B, best_state_B, best_val_f1_B = train_model(
        model=model_B,
        learning_rate=1e-5,
        epochs=30,
        optimizer_type=optimizer_type,
        state_dict_path = Save_path_B,
        Save_path=Save_path_B,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        weight_decay=weight_decay,
        momentum=momentum,
        history=history_B,
        best_val_f1=best_val_f1_B
    )

    plot_path_A = Save_path_A.replace('.pth', '.png')
    plot_path_B = Save_path_B.replace('.pth', '.png')
    
    plot_loss_curves(history_A, model_A, optimizer_type=optimizer_type, save_path=plot_path_A, phase_boundaries=[20])
    plot_loss_curves(history_B, model_B, optimizer_type=optimizer_type, save_path=plot_path_B, phase_boundaries=[20])

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, (mean, std) = get_dataloaders(batch_size= 64, num_workers= 2)

    
    #Using Adam optimizer for both models
    optimizer_type = "adam"

    model_A = NodeConvs_Net(in_channels=3, base_channels= 32, levels= 3, dropout= 0.4, fc_depth= 1).to(device)
    print(f"Total trainable parameters of Model A: {model_A.count_trainable_parameters():,}")
    model_B = NodeConvs_B_Net(in_channels=3, base_channels= 32, levels= 3, dropout= 0.4, fc_depth= 1).to(device)
    print(f"Total trainable parameters of Model B: {model_B.count_trainable_parameters():,}")
    print(f"Training using optimizer: {optimizer_type}")
    model_comparison(
        model_A=model_A,
        model_B=model_B,
        optimizer_type=optimizer_type,
        Save_path_A="model_A.pth",
        Save_path_B="model_B.pth",
        train_loader=train_loader,
        val_loader=val_loader
    )

    optimizer_type = "sgd"
    model_A = NodeConvs_Net(in_channels=3, base_channels= 32, levels= 3, dropout= 0.4, fc_depth= 1).to(device)
    print(f"Total trainable parameters of Model A: {model_A.count_trainable_parameters():,}")
    model_B = NodeConvs_B_Net(in_channels=3, base_channels= 32, levels= 3, dropout= 0.4, fc_depth= 1).to(device)
    print(f"Total trainable parameters of Model B: {model_B.count_trainable_parameters():,}")
    print(f"Training using optimizer: {optimizer_type}")
    model_comparison(
        model_A=model_A,
        model_B=model_B,
        optimizer_type=optimizer_type,
        Save_path_A="model_A_sgd.pth",
        Save_path_B="model_B_sgd.pth",
        train_loader=train_loader,
        val_loader=val_loader
    )   

    optimizer_type = "sgd_momentum"
    model_A = NodeConvs_Net(in_channels=3, base_channels= 32, levels= 3, dropout= 0.4, fc_depth= 1).to(device)
    print(f"Total trainable parameters of Model A: {model_A.count_trainable_parameters():,}")
    model_B = NodeConvs_B_Net(in_channels=3, base_channels= 32, levels= 3, dropout= 0.4, fc_depth= 1).to(device)
    print(f"Total trainable parameters of Model B: {model_B.count_trainable_parameters():,}")
    print(f"Training using optimizer: {optimizer_type}")
    model_comparison(
        model_A=model_A,
        model_B=model_B,
        optimizer_type=optimizer_type,
        Save_path_A="model_A_sgd_momentum.pth",
        Save_path_B="model_B_sgd_momentum.pth",
        train_loader=train_loader,
        val_loader=val_loader
    )   
