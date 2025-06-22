import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np

def get_resnet18_encoder(n_input_channels=2, device='cpu'):
    """
    Returns a modified ResNet-18 encoder that accepts n_input_channels.
    The first conv layer is adapted, with pretrained weights migrated by channel mean.
    """
    model = models.resnet18(pretrained=True)
    pretrained_w = model.conv1.weight.data  # (64, 3, 7, 7)
    new_conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Use the mean of pretrained RGB weights for all new input channels
    mean_weight = pretrained_w.mean(dim=1, keepdim=True)
    for i in range(n_input_channels):
        new_conv1.weight.data[:, i, :, :] = mean_weight[:, 0, :, :]
    model.conv1 = new_conv1
    encoder = nn.Sequential(*list(model.children())[:-1])
    encoder = encoder.to(device)
    encoder.eval()
    return encoder

def extract_features(img_paths, encoder, device='cpu', verbose=100):
    """
    Supports both 2D ([2,256,256]) and 2.5D ([2,N,256,256]) npy files.
    Will auto-detect shape and adjust accordingly.
    """
    features = []
    error_files = []
    n_input_channels = None  # To auto-detect on first run

    for i, path in enumerate(img_paths):
        try:
            arr = np.load(path)  # (2,256,256) or (2,N,256,256)
            # --- Auto-detect input channels ---
            if arr.ndim == 4:
                # 2.5D: (2, N, 256, 256) → (2*N, 256, 256)
                arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
            elif arr.shape == (2, 256, 256):
                # 2D: do nothing
                pass
            else:
                print(f"[WARNING] Shape error for {path}: {arr.shape}")
                error_files.append(str(path))
                continue
            # On first valid sample, set expected channel count
            if n_input_channels is None:
                n_input_channels = arr.shape[0]
                # Optional: Check encoder first conv layer matches input channels
                conv_in = list(encoder.children())[0].in_channels
                if n_input_channels != conv_in:
                    print(f"[ERROR] Input channels ({n_input_channels}) != encoder in_channels ({conv_in})")
                    error_files.append(str(path))
                    continue
            # Shape check
            if arr.shape[0] != n_input_channels:
                print(f"[WARNING] Channel mismatch for {path}: {arr.shape[0]} vs expected {n_input_channels}")
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
