import timm
import torch
import torch.nn as nn


class MultiheadClassifier(nn.Module):
    def __init__(self, backbone_name, backbone_pretrained, input_size, num_heads, num_classes):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, backbone_pretrained, num_classes=0)
        backbone_out_features_num = self.backbone(torch.randn(1, 3, input_size[1], input_size[0])).size(1)

        self.heads = nn.ModuleList([
            nn.Linear(backbone_out_features_num, num_classes) for _ in range(num_heads)
        ])
    
    def forward(self, x):
        features = self.backbone(x)
        logits = [head(features) for head in self.heads]
        return logits