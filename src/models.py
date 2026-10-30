import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Compact U-Net for binary tumor segmentation (~7.8M params at base=32)."""

    def __init__(self, in_ch=1, base=32, depth=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.downs.append(ConvBlock(c, ch))
            c = ch
        self.bottleneck = ConvBlock(chs[-2], chs[-1])
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for ch in reversed(chs[:-1]):
            self.ups.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.up_convs.append(ConvBlock(ch * 2, ch))
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, conv, s in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = conv(torch.cat([x, s], 1))
        return self.head(x)


def size_classifier(num_classes=3, pretrained=True):
    """Baseline: ResNet-18 predicting the size category directly from the image (no segmentation)."""
    import torchvision
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    w = m.conv1.weight.data.sum(1, keepdim=True)  # 3-channel -> 1-channel, keep pretrained filters
    m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    m.conv1.weight.data = w
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def dice_bce_loss(logits, target, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)
    return bce + (1 - dice).mean()
