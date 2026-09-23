from NodeConvs_Net import NodeConvs_Net
from Data_loader import get_dataloaders
from Train import compute_eval_metrics
import torch
import torch.nn as nn
import numpy as np
import time
import copy
import torch.nn.functional as F
from sklearn.metrics import classification_report


train_loader, val_loader, test_loader, (mean, std) = get_dataloaders(batch_size= 64, num_workers= 2)


def evaluate():
    test_probs_list = []
    test_targets_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = NodeConvs_Net(in_channels=3, base_channels= 32, levels= 3, dropout= 0.4, fc_depth= 1).to(device)
    print(f"Total trainable parameters: {model.count_trainable_parameters():,}")

    state_dict = torch.load('Weights/Best_modelA_1.pth', weights_only=True)
    model.load_state_dict(state_dict)
    print("loaded the best saved model")

    model.eval()
    start_time = time.time()

    with torch.inference_mode():
        for images, labels in test_loader:
            images , labels = images.to(device), labels.to(device)

            pred_logits = model(images)

            probs = F.softmax(pred_logits, dim=1)
            test_loader_probs_list.append(probs.cpu().numpy())
            test_targets_list.append(labels.cpu().numpy())

    end_time = time.time()
    test_probs = np.concatenate(test_probs_list, axis=0)
    test_targets = np.concatenate(test_targets_list, axis=0)
    test_preds = np.argmax(test_probs, axis=1)

    test_metrics = compute_eval_metrics(test_targets, test_probs)
    inference_time = end_time - start_time
    throughput = len(test_targets) / inference_time

    # ==========================================
    # FORMATTED PRINT STATEMENTS
    # ==========================================
    print("="*50)
    print(" 🩸 BLOODMNIST TEST SET EVALUATION")
    print("="*50)
    print(f"Total Test Samples : {len(test_targets)}")
    print(f"Inference Time     : {inference_time:.2f} seconds")
    print(f"Throughput         : {throughput:.0f} images/sec")
    print("-" * 50)
    print(f"Overall Accuracy   : {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro ROC-AUC      : {test_metrics['auc']:.4f}")
    print(f"Macro F1-Score     : {test_metrics['f1']:.4f}")
    print(f"Macro Precision    : {test_metrics['precision']:.4f}")
    print(f"Macro Recall       : {test_metrics['recall']:.4f}")
    print("="*50)
    
    print("\n[PER-CLASS CLASSIFICATION REPORT]")
    # The classification report provides a breakdown for every single class
    print(classification_report(test_targets, test_preds, digits=4))

    print("\n[CONFUSION MATRIX]")
    # Formats the numpy array slightly nicer for terminal output
    cm = test_metrics['confusion_matrix']
    for row in cm:
        print(" ".join(f"{val:4d}" for val in row))


if __name__ == "__main__":
    evaluate()
