import torch
import torch.nn as nn
from transformers import Dinov2ForImageClassification
from wavelet import HaarWaveletTransform

class WaveletDINO(nn.Module):
    def __init__(self, num_classes=2, use_all_bands=True):
        super().__init__()
        self.wavelet = HaarWaveletTransform()
        self.use_all_bands = use_all_bands
        self.backbone = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Linear(in_features * (4 if use_all_bands else 1), num_classes)

    def forward(self, x):
        Yl, xH, xV, xD = self.wavelet(x)
        feat_Yl = self.backbone(Yl).logits
        if self.use_all_bands:
            feat_xH = self.backbone(xH).logits
            feat_xV = self.backbone(xV).logits
            feat_xD = self.backbone(xD).logits
            features = torch.cat([feat_Yl, feat_xH, feat_xV, feat_xD], dim=1)
        else:
            features = feat_Yl
        return self.classifier(features)
