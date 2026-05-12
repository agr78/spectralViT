import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from scipy.interpolate import PchipInterpolator

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