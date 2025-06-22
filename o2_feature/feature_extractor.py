# o2_feature/feature_extractor.py

import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np

def get_resnet18_encoder(device='cpu'):
    model = models.resnet18(pretrained=True)
    pretrained_w = model.conv1.weight.data
    new_conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    mean_weight = pretrained_w.mean(dim=1, keepdim=True)
    new_conv1.weight.data[:, 0, :, :] = mean_weight[:, 0, :, :]
    new_conv1.weight.data[:, 1, :, :] = mean_weight[:, 0, :, :]
    model.conv1 = new_conv1
    encoder = nn.Sequential(*list(model.children())[:-1])
    encoder = encoder.to(device)
    encoder.eval()
    return encoder

def extract_features(img_paths, encoder, device='cpu', verbose=100):
    features = []
    error_files = []
    for i, path in enumerate(img_paths):
        try:
            arr = np.load(path)  # shape: [2,256,256]
            if arr.shape != (2, 256, 256):
                print(f"[WARNING] Shape error for {path}: {arr.shape}")
                error_files.append(str(path))
                continue
            img_tensor = torch.tensor(arr.copy(), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = encoder(img_tensor)
                feat = feat.view(-1).cpu().numpy()
            features.append(feat)
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            error_files.append(str(path))
        if (i + 1) % verbose == 0:
            print(f"Extracted features for {i + 1}/{len(img_paths)} slices")
    if features:
        features = np.vstack(features)
    else:
        features = np.empty((0, 512))
    return features, error_files
