import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import roc_auc_score, confusion_matrix


def mask_crop(data, mask, pad, viz=False):
    """Crop data using mask with optional padding."""
    z_idx = ~(mask == 0).all((0, 1))
    x_idx = ~(mask == 0).all((1, 2))
    y_idx = ~(mask == 0).all((0, 2))
    pad_x = pad[0]
    pad_y = pad[1]
    pad_z = pad[2]
    if viz:
        print('Cropping:', x_idx, y_idx, z_idx)
    cropped_mask = mask[:, :, ~(mask == 0).all((0, 1))]
    cropped_mask = cropped_mask[~(mask == 0).all((1, 2)), :, :]
    cropped_mask = cropped_mask[:, ~(mask == 0).all((0, 2)), :]
    if data.shape != mask.shape:
        data = data[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
    img = data[:, :, ~(mask == 0).all((0, 1))]
    img = img[~(mask == 0).all((1, 2)), :, :]
    img = img[:, ~(mask == 0).all((0, 2)), :]
    if np.sum(pad) != 0:
        x0 = ~(mask == 0).all((1, 2))
        x00 = np.where(x0)[0][0]
        x0f = np.where(x0)[0][-1]
        y0 = ~(mask == 0).all((0, 2))
        y00 = np.where(y0)[0][0]
        y0f = np.where(y0)[0][-1]
        z0 = ~(mask == 0).all((0, 1))
        z00 = np.where(z0)[0][0]
        z0f = np.where(z0)[0][-1]
        img = data[x00:x0f, y00:y0f, z00:z0f]
        x = (pad_x - img.shape[0]) / 2
        y = (pad_y - img.shape[1]) / 2
        z = (pad_z - img.shape[2]) / 2
        img = data[int(x00 - x):int(x0f + x), int(y00 - y):int(y0f + y), int(z00 - z):int(z0f + z)]
    if viz:
        print(img.shape)
        print(cropped_mask.shape)
        plt.imshow(np.rot90(img[:, :, img.shape[2] // 2]))
        plt.show()
    return img

def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def smooth_curve(x, y):
    x, y = np.array(x), np.array(y)
    if len(x) <= 2: return x, y
    x_log = np.log10(x)
    x_new_log = np.linspace(x_log.min(), x_log.max(), 1000)
    interp = PchipInterpolator(x_log, y)
    y_smooth = interp(x_new_log)
    return 10**x_new_log, np.clip(y_smooth, 0, 1.0)

def plot_all_model_losses(spectral_vit_history, 
                          spatial_vit_history, 
                          compact_spatial_vit_history, 
                          unet_history, swin_history,
                          EPOCHS):
    combined = {**spectral_vit_history, **spatial_vit_history, **compact_spatial_vit_history, **unet_history, **swin_history}
    plt.figure(figsize=(12, 7))
    
    configs = [
            ('spectral_vit_loss', 'Spectral ViT', 'tab:blue', '-'),
            ('spatial_vit_loss', 'Spatial ViT (Heavy)', 'tab:orange', '--'), # Check this key too!
            ('compact_spatial_vit_loss', 'Spatial ViT (Matched)', 'tab:green', '--'),
            ('swin_loss', 'Swin ViT', 'tab:purple', '-.'),
            ('unet_loss', 'Attn U-Net', 'tab:red', ':'),
        ]
        
    for key, label, color, style in configs:
        if key in combined and len(combined[key]) > 0:
            data = np.array(combined[key])
            mean_loss = np.mean(data, axis=0)
            std_loss = np.std(data, axis=0)
            epochs = range(1, len(mean_loss) + 1)
            
            plt.plot(epochs, mean_loss, label=label, color=color, linestyle=style, lw=2)
            plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.1)

    plt.title("Cross-Validated Training Loss: Model Comparison", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.xlim([1, EPOCHS])
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def get_metrics_array(y_true, y_probs):
    """Calculate metrics for a set of predictions."""
    preds = (y_probs > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    acc = (tp + tn) / len(y_true)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(y_true, y_probs)
    return np.array([acc, sens, spec, auc])