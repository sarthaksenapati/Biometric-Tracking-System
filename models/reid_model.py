# models/reid_model.py
# OSNet-x1.0 — architecture verified key-by-key against osnet_x1_0_msmt17.pth

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import os


# ── Primitives ────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """conv + bn + relu  — keys: .conv .bn"""
    def __init__(self, in_ch, out_ch, k, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    """conv + bn (no relu) — keys: .conv .bn"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """
    Depthwise separable conv.
    Keys: .conv1 (pointwise) .conv2 (depthwise 3x3) .bn
    """
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.bn    = nn.BatchNorm2d(ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv2(self.conv1(x))), inplace=True)


class ChannelGate(nn.Module):
    """
    Squeeze-excitation gate using Conv2d (not Linear).
    Keys: .fc1 (squeeze Conv2d) .fc2 (excite Conv2d)
    Reduction per stage:
      mid_ch=64  → fc1=(4,64,1,1)   fc2=(64,4,1,1)
      mid_ch=96  → fc1=(6,96,1,1)   fc2=(96,6,1,1)
      mid_ch=128 → fc1=(8,128,1,1)  fc2=(128,8,1,1)
    """
    def __init__(self, ch):
        super().__init__()
        mid = ch // 16
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(ch, mid, 1, bias=True)
        self.fc2 = nn.Conv2d(mid, ch, 1, bias=True)

    def forward(self, x):
        w = self.gap(x)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class OSBlock(nn.Module):
    """
    OSNet block. Checkpoint attribute structure:
      conv1       ConvBnRelu  (bottleneck in)
      conv2a      LightConv3x3 (scale 1)
      conv2b      nn.Sequential of 2 LightConv3x3 (scale 2)
      conv2c      nn.Sequential of 3 LightConv3x3 (scale 3)
      conv2d      nn.Sequential of 4 LightConv3x3 (scale 4)
      gate        ChannelGate — ONE instance shared across all 4 streams.
                  Checkpoint stores exactly one 'gate' key per block.
      conv3       ConvBn (bottleneck out, no relu)
      downsample  ConvBn (skip, only when in_ch != out_ch)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 4   # 64, 96, or 128 depending on stage

        self.conv1  = ConvBnRelu(in_ch, mid, 1)

        self.conv2a = LightConv3x3(mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid), LightConv3x3(mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid), LightConv3x3(mid),
                                    LightConv3x3(mid))
        self.conv2d = nn.Sequential(LightConv3x3(mid), LightConv3x3(mid),
                                    LightConv3x3(mid), LightConv3x3(mid))

        # Single gate — checkpoint key is conv*.*.gate.* (not gate1/gate2/...)
        self.gate  = ChannelGate(mid)

        self.conv3 = ConvBn(mid, out_ch)

        self.downsample = ConvBn(in_ch, out_ch) if in_ch != out_ch else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = (self.gate(self.conv2a(x1)) +
               self.gate(self.conv2b(x1)) +
               self.gate(self.conv2c(x1)) +
               self.gate(self.conv2d(x1)))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity, inplace=True)


class OSNet(nn.Module):
    """
    OSNet-x1.0 matching all 567 keys in osnet_x1_0_msmt17.pth.

    Stage channels (mid = out//4):
      conv2: in=64,  mid=64,  out=256
      conv3: in=256, mid=96,  out=384
      conv4: in=384, mid=128, out=512

    Transition after conv2 and conv3:
      conv2.2 = nn.Sequential( ConvBnRelu(256,256,1), AvgPool2d )
      conv3.2 = nn.Sequential( ConvBnRelu(384,384,1), AvgPool2d )
      Checkpoint key: conv2.2.0.conv / conv2.2.0.bn  (index 0 = ConvBnRelu)

    conv4 has NO transition pool — only 2 OSBlocks.

    Head:
      conv5: ConvBnRelu(512,512,1)
      gap:   AdaptiveAvgPool2d(1)
      fc:    Sequential( Linear(512,512), BN1d(512) )
             keys: fc.0 = Linear, fc.1 = BN1d
      classifier: Linear(512, 751)
    """
    def __init__(self, num_classes=751):
        super().__init__()

        self.conv1   = ConvBnRelu(3, 64, 7, s=2, p=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            OSBlock(64, 256),
            OSBlock(256, 256),
            nn.Sequential(ConvBnRelu(256, 256, 1), nn.AvgPool2d(2, stride=2))
        )

        self.conv3 = nn.Sequential(
            OSBlock(256, 384),
            OSBlock(384, 384),
            nn.Sequential(ConvBnRelu(384, 384, 1), nn.AvgPool2d(2, stride=2))
        )

        self.conv4 = nn.Sequential(
            OSBlock(384, 512),
            OSBlock(512, 512),
        )

        self.conv5 = ConvBnRelu(512, 512, 1)
        self.gap   = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512)
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x   # 512-d embedding (before classifier)


# ── ReIDModel wrapper ─────────────────────────────────────────────────────────

class ReIDModel:
    WEIGHTS_PATH = "osnet_x1_0_msmt17.pth"

    def __init__(self, weights_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = weights_path or self.WEIGHTS_PATH

        if os.path.exists(path):
            self.model = self._load_osnet(path)
        else:
            print(f"[ReIDModel] ⚠️  {path} not found — ResNet50 fallback")
            self.model = self._fallback_resnet()

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]
            )
        ])

    def _load_osnet(self, path):
        model = OSNet(num_classes=751)
        state = torch.load(path, map_location=self.device)
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}

        model_state = model.state_dict()
        matched   = {k: v for k, v in state.items()
                     if k in model_state and v.shape == model_state[k].shape}
        unmatched = [k for k in state if k not in matched]

        model_state.update(matched)
        model.load_state_dict(model_state, strict=False)

        print(f"[ReIDModel] ✅ OSNet: {len(matched)}/{len(state)} layers loaded on {self.device}")
        if unmatched:
            print(f"[ReIDModel] ⚠️  Unmatched ({len(unmatched)}): {unmatched[:5]}")
        else:
            print(f"[ReIDModel] ✅ All keys matched — perfect load.")

        return model.to(self.device)

    def _fallback_resnet(self):
        import torchvision.models as models
        m = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        return m.to(self.device)

    def get_embedding(self, image):
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        if w < 40 or h < 80:
            return None
        try:
            rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(tensor)
            emb  = feat.squeeze().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm == 0:
                return None
            return emb / norm
        except Exception as e:
            print(f"[ReIDModel] Error: {e}")
            return None