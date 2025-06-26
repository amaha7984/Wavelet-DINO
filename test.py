import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import Dinov2ForImageClassification
from sklearn.metrics import (
    precision_score, average_precision_score, roc_auc_score,
    recall_score, f1_score
)
import numpy as np
import ptwt
import pywt
import argparse

# ======================== Haar Wavelet Transform ========================
class HaarTransform(nn.Module):
    def __init__(self, level=1, mode="symmetric", with_grad=False):
        super().__init__()
        self.wavelet = pywt.Wavelet("haar")
        self.level = level
        self.mode = mode
        self.with_grad = with_grad

    def forward(self, x):
        with torch.set_grad_enabled(self.with_grad):
            Yl, *Yh = ptwt.wavedec2(x.float(), wavelet=self.wavelet, level=self.level, mode=self.mode)
            xH, xV, xD = Yh[0]

            Yl = F.interpolate(Yl, size=(224, 224), mode='bilinear', align_corners=False)
            xH = F.interpolate(xH, size=(224, 224), mode='bilinear', align_corners=False)
            xV = F.interpolate(xV, size=(224, 224), mode='bilinear', align_corners=False)
            xD = F.interpolate(xD, size=(224, 224), mode='bilinear', align_corners=False)
            return Yl, xH, xV, xD

# ======================== WaveletDINO Model ========================
class WaveletDINO(nn.Module):
    def __init__(self, num_classes=2, use_all_bands=True):
        super().__init__()
        self.dwt = HaarTransform()
        self.use_all_bands = use_all_bands
        self.backbone = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Linear(in_features * (4 if use_all_bands else 1), num_classes)

    def forward(self, x):
        Yl, xH, xV, xD = self.dwt(x)
        feat_Yl = self.backbone(Yl).logits

        if self.use_all_bands:
            feat_xH = self.backbone(xH).logits
            feat_xV = self.backbone(xV).logits
            feat_xD = self.backbone(xD).logits
            features = torch.cat([feat_Yl, feat_xH, feat_xV, feat_xD], dim=1)
        else:
            features = feat_Yl

        return self.classifier(features)

# ======================== Evaluation Logic ========================
def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_probs = []
    correct_real, correct_fake = 0, 0
    total_real, total_fake = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs > 0.5).int()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total_real += (labels == 0).sum().item()
            total_fake += (labels == 1).sum().item()
            correct_real += ((preds == 0) & (labels == 0)).sum().item()
            correct_fake += ((preds == 1) & (labels == 1)).sum().item()

    y_true = np.array(all_labels)
    y_score = np.array(all_probs)
    y_pred = (y_score > 0.5).astype(int)

    precision_real = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    precision_fake = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_real = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_fake = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_real = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_fake = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    accuracy_real = 100.0 * correct_real / total_real if total_real else 0.0
    accuracy_fake = 100.0 * correct_fake / total_fake if total_fake else 0.0
    overall_accuracy = 100.0 * (correct_real + correct_fake) / (total_real + total_fake)
    average_precision = average_precision_score(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    return {
        "Overall Accuracy": overall_accuracy,
        "Accuracy (Real)": accuracy_real,
        "Accuracy (Fake)": accuracy_fake,
        "Precision (Real)": precision_real,
        "Precision (Fake)": precision_fake,
        "Recall (Real)": recall_real,
        "Recall (Fake)": recall_fake,
        "F1-score (Real)": f1_real,
        "F1-score (Fake)": f1_fake,
        "Average Precision": average_precision,
        "AUC": auc
    }

# ======================== Main Runner ========================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = datasets.ImageFolder(root=args.test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = WaveletDINO(num_classes=args.num_classes, use_all_bands=args.use_all_bands)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)

    metrics = evaluate(model, test_loader, device)

    print(f"\n--- Evaluation Results for: {os.path.basename(args.test_path)} (All bands: {args.use_all_bands}) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

# ======================== Argument Parser ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate WaveletDINO on test data")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test dataset (ImageFolder format)")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to trained model (.pth)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--use_all_bands", action="store_true", help="Use all wavelet subbands (Yl + H, V, D)")
    args = parser.parse_args()

    main(args)
