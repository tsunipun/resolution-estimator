import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class ResolutionEstimator(nn.Module):
    def __init__(self, pretrained=False):
        super(ResolutionEstimator, self).__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        
        # Replace the classifier head for regression (1 output)
        # MobileNetV3-Small classifier structure:
        # Sequential(
        #   (0): Linear(in_features=576, out_features=1024, bias=True)
        #   (1): Hardswish()
        #   (2): Dropout(p=0.2, inplace=True)
        #   (3): Linear(in_features=1024, out_features=1000, bias=True)
        # )
        
        # We keep the first linear layer but change the last one.
        # Or simpler: just replace self.backbone.classifier with our own.
        
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid() # Output 0.0 to 1.0
        )

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    model = ResolutionEstimator()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Output value: {y.item()}")
