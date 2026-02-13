import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import numpy as np
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict, Counter
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import pandas as pd

def mask_crop(data,mask,pad,viz=False):
    z_idx = ~(mask==0).all((0,1))
    x_idx = ~(mask==0).all((1,2))
    y_idx = ~(mask==0).all((0,2))
    pad_x = pad[0]
    pad_y = pad[1]
    pad_z = pad[2]
    # print('Cropping:',x_idx,y_idx,z_idx)
    cropped_mask = mask[:,:,~(mask==0).all((0,1))]
    cropped_mask = cropped_mask[~(mask==0).all((1,2)),:,:]
    cropped_mask = cropped_mask[:,~(mask==0).all((0,2)),:]
    if data.shape != mask.shape:
        data = data[:mask.shape[0],:mask.shape[1],:mask.shape[2]]
    img = data[:,:,~(mask==0).all((0,1))]
    img = img[~(mask==0).all((1,2)),:,:]
    img = img[:,~(mask==0).all((0,2)),:]
    if np.sum(pad) != 0:
        x0 = ~(mask==0).all((1,2))
        x00 = np.where(x0)[0][0]
        x0f = np.where(x0)[0][-1]
        y0 = ~(mask==0).all((0,2))
        y00 = np.where(y0)[0][0]
        y0f = np.where(y0)[0][-1]
        z0 = ~(mask==0).all((0,1))
        z00 = np.where(z0)[0][0]
        z0f = np.where(z0)[0][-1]
        img = data[x00:x0f,y00:y0f,z00:z0f]
        x = (pad_x-img.shape[0])/2
        y = (pad_y-img.shape[1])/2
        z = (pad_z-img.shape[2])/2
        img = data[int(x00-x):int(x0f+x),int(y00-y):int(y0f+y),int(z00-z):int(z0f+z)]
    if viz:
        print(img.shape)
        print(cropped_mask.shape)
        plt.imshow(np.rot90(img[:,:,img.shape[2]//2]))
        plt.show()
    return img

class AugmentPipe(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p  # initial probability
        self.augment_ops = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            #transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        ])

    def forward(self, x):
        if torch.rand(1).item() < self.p:
            # Assume x is a flattened image, reshape before applying augmentations
            b, dim = x.shape
            x_img = x.view(b, 1, 28, 28)
            x_img = self.augment_ops(x_img)
            return x_img.view(b, -1)
        else:
            return x

def class_weights_calc(train_subset,num_classes):
    # Compute class weights for weighted cross-entropy loss
    train_labels = [label for _, label in train_subset]
    label_counts = Counter(train_labels)
    total_count = sum(label_counts.values())
    class_weights = []
    for c in range(num_classes):
        count = label_counts[c] if c in label_counts else 0
        class_weights.append(total_count / (count + 1e-8))  # epsilon to avoid division by zero
    return class_weights


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Stratified Training-validation split
def stratified_split(dataset, val_ratio=0.2, class_labels=[0,1]):
    label_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label in class_labels:
            label_to_indices[label].append(i)

    train_indices, val_indices = [], []
    for label, inds in label_to_indices.items():
        inds = torch.tensor(inds)
        n_total = len(inds)
        n_val = int(n_total * val_ratio)
        perm = torch.randperm(n_total)
        val_indices += inds[perm[:n_val]].tolist()
        train_indices += inds[perm[n_val:]].tolist()

    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def gradient_penalty(D, real_x, real_y_emb, fake_x, fake_y_emb):
    bs = real_x.size(0)
    alpha = torch.rand(bs, 1, 1, 1, device=real_x.device)
    interpolated_x = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)
    alpha_y = alpha.view(bs, 1)
    interpolated_y = (alpha_y * real_y_emb + (1 - alpha_y) * fake_y_emb).requires_grad_(True)
    d_interpolated = D(interpolated_x, interpolated_y)
    grads = torch.autograd.grad(outputs=d_interpolated, inputs=[interpolated_x, interpolated_y],
                                grad_outputs=torch.ones_like(d_interpolated), create_graph=True,
                                retain_graph=True, only_inputs=True)
    grad_x, grad_y = grads
    grad_x = grad_x.view(bs, -1)
    grad_y = grad_y.view(bs, -1)
    grad_norm = torch.sqrt(grad_x.pow(2).sum(1) + grad_y.pow(2).sum(1) + 1e-12)
    return ((grad_norm - 1) ** 2).mean()

def get_timestep_embedding(timesteps, dim=128):
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

def get_label_embedding(self, y):
    one_hot = F.one_hot(y, num_classes=self.num_classes).float().to(self.label_embed.weight.device)
    cond = self.label_embed(one_hot)
    return cond

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def q_sample(x_start, t, alphas, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    sqrt_alpha = sqrt_alphas_cumprod[t][:, None, None, None]
    sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

def q_label_sample(x_start, t, alphas, noise=None):
    """
    Diffuse labels at timestep t using the standard DDPM formula,
    while preserving the original label shape to avoid broadcasting issues.
    
    Args:
        x_start: [B, 1] tensor of original labels
        t: [B] tensor of timesteps
        alphas: [T] tensor of alpha values
        noise: optional noise tensor of shape [B, 1]
    
    Returns:
        Noisy labels [B, 1]
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    # Preserve shape [B,1] to match predictions
    sqrt_alpha = sqrt_alphas_cumprod[t].view(-1, 1)
    sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

    return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise


@torch.no_grad()
def sample_images(
    model,
    timesteps,
    device,
    image_size,
    chs,
    num_samples=8,
    class_label=0,
    betas=None,     # NEW (optional)
    alphas=None,    # NEW (optional)
    alphas_cumprod=None  # NEW (optional)
):
    # --- If not provided, fall back to old behavior ---
    if betas is None or alphas is None or alphas_cumprod is None:
        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    model.eval()
    x = torch.randn(num_samples, chs, image_size, image_size).to(device)

    y = torch.full((num_samples,), class_label, dtype=torch.long, device=device)
    cond = model.get_label_embedding(y)

    for i in reversed(range(timesteps)):
        t_i = torch.full((num_samples,), i, device=device, dtype=torch.long)
        noise_pred, _ = model(x, t_i, cond)
        alpha = alphas[i]
        alpha_bar = alphas_cumprod[i]
        beta = betas[i]

        noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
        x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred) + torch.sqrt(beta) * noise

    return x.clamp(-1, 1)

# def sample_images(model, timesteps, device, image_size, chs, num_samples=8, class_label=0):
#     betas = cosine_beta_schedule(timesteps).to(device)
#     alphas = 1. - betas
#     alphas_cumprod = torch.cumprod(alphas, dim=0)
#     sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
#     sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
#     model.eval()
#     x = torch.randn(num_samples, chs, image_size, image_size).to(device)
#     t = torch.full((num_samples,), timesteps - 1, device=device, dtype=torch.long)

#     # Get conditional embeddings for the target class
#     y = torch.full((num_samples,), class_label, dtype=torch.long, device=device)
#     cond = model.get_label_embedding(y)

#     for i in reversed(range(timesteps)):
#         t_i = torch.full((num_samples,), i, device=device, dtype=torch.long)
#         noise_pred, _ = model(x, t_i, cond)
#         alpha = alphas[i]
#         alpha_bar = alphas_cumprod[i]
#         beta = betas[i]

#         if i > 0:
#             noise = torch.randn_like(x)
#         else:
#             noise = torch.zeros_like(x)

#         x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred) + torch.sqrt(beta) * noise

#     return x.clamp(-1, 1)

import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def calc_metrics(y_true, y_probs, threshold=0.5):
    # Convert probabilities to binary predictions based on the threshold
    y_pred = (np.array(y_probs) >= threshold).astype(int)
    y_true = np.array(y_true)

    acc = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    
    # Handle the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return acc, precision, recall, specificity
def flatten_data(loader):
    all_x = []
    all_y = []
    
    # We iterate through the loader to respect the Sampler's balancing
    for images, clinical, labels in loader:
        # Flatten: (Batch, Ch, H, W) -> (Batch, Ch*H*W)
        x_flat = images.view(images.size(0), -1).numpy()
        all_x.append(x_flat)
        all_y.append(labels.numpy())
    
    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)

def filter_data(file_path,cv_names,filter_data):
    df = pd.read_csv(file_path)
    dfd = df.copy()
    if filter_data == True:
        # Drop blank columns
        try:
            for (columnName, columnData) in dfd.items():
                if columnData.isnull().all():
                    print('Dropping NaN column at',columnName)
                    dfd.drop(columnName,axis=1,inplace=True)
            # Add relevant column names from headers
            for (columnName, columnData) in dfd.items():
                    dfd.rename(columns={columnName:columnName+': '+columnData.values[0]},inplace=True)
        except:
            for (columnName, columnData) in dfd.items():
                if columnData.isnull().all():
                    print('Dropping NaN column at',columnName)
                    dfd.drop(columnName,axis=1,inplace=True)
            # Add relevant column names from headers
            for (columnName, columnData) in dfd.items():
                    dfd.rename(columns={columnName:columnName+': '+columnData.values[0]},inplace=True)
        def drop_prefix(self, prefix):
            self.columns = self.columns.str.lstrip(prefix)
            return self
        pd.core.frame.DataFrame.drop_prefix = drop_prefix
        dfd.drop_prefix('Unnamed:')
        motor_df = dfd.copy()      
        try:  
            for (columnName, columnData) in motor_df.items():
                if columnName[1].isdigit():
                    motor_df.rename(columns={columnName:columnName[4:]},inplace=True)
        except:
            for (columnName, columnData) in motor_df.items():
                if columnName[1].isdigit():
                    motor_df.rename(columns={columnName:columnName[4:]},inplace=True)
        # Drop non-motor (III) columns
        for (columnName, columnData) in motor_df.items():
            if columnName in cv_names:
                print('Keeping',columnName)
                next
            else:
                motor_df.drop(columnName,axis=1,inplace=True)
        # Drop subheader
        motor_df = motor_df.tail(-1)
        motor_df = motor_df.replace('na',np.nan)
        motor_df = motor_df.dropna()
    else:
        motor_df = dfd
    return motor_df
