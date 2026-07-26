import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import random
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, f1_score, accuracy_score, top_k_accuracy_score, roc_auc_score
import scipy.stats as stats
import numpy as np
from networks import pca_tokenize, fourier_tokenize, laplacian_tokenize

def robust_flatten(img_np, target_dim):
    t = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(t, size=(target_dim, target_dim), mode='bilinear', align_corners=False)
    return resized.numpy().flatten()

def compute_comprehensive_metrics(y_true, y_prob, threshold):
    """Calculates performance metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        'AUC': roc_auc_score(y_true, y_prob),
        'B-Acc': (sens + spec) / 2,
        'Spec': spec,
        'F1': f1_score(y_true, y_pred, zero_division=0)
    }

def mean_std_str(values):
    values = np.asarray(values)
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}"

def delong_roc_variance(ground_truth, predictions):
    """
    Computes the structural components (V10 and V01) for DeLong's variance.
    """
    m = sum(ground_truth == 1)
    n = sum(ground_truth == 0)
    
    # Get scores of positives and negatives
    pos = predictions[ground_truth == 1]
    neg = predictions[ground_truth == 0]
    
    # Compute structural components (V10 and V01)
    # v10[i] captures how often negative samples score lower than the i-th positive sample
    v10 = np.array([np.sum(neg < p) + 0.5 * np.sum(neg == p) for p in pos]) / n
    # v01[j] captures how often positive samples score higher than the j-th negative sample
    v01 = np.array([np.sum(pos > p) + 0.5 * np.sum(pos == p) for p in neg]) / m
    
    return v10, v01

def compare_auc_significance(y_true, prob_base, prob_model):
    """
    Exact implementation of the DeLong test for two correlated ROC curves.
    """
    y_true = np.array(y_true)
    prob_base = np.array(prob_base)
    prob_model = np.array(prob_model)
    
    auc_base = roc_auc_score(y_true, prob_base)
    auc_model = roc_auc_score(y_true, prob_model)
    
    # Get structural components
    v10_base, v01_base = delong_roc_variance(y_true, prob_base)
    v10_model, v01_model = delong_roc_variance(y_true, prob_model)
    
    # Covariance matrices of the structural components
    S10 = np.cov(v10_base, v10_model)
    S01 = np.cov(v01_base, v01_model)
    
    m = sum(y_true == 1)
    n = sum(y_true == 0)
    
    # DeLong variance of the difference (accounting for covariance)
    var_diff = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + \
               (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n
               
    # Z-score and two-sided p-value
    z = (auc_model - auc_base) / np.sqrt(max(var_diff, 1e-8))
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return auc_model - auc_base, p_value

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

def get_cv_fold_metrics(name,cv_probs,all_cv_labels,thresholds):
    probs = np.array(cv_probs[name])
    labels = np.array(all_cv_labels)
    fold_indices = np.array_split(np.arange(len(labels)), 5)

    fold_results = []
    for idxs in fold_indices:
        m = compute_comprehensive_metrics(labels[idxs], probs[idxs], thresholds[name])
        fold_results.append(m)
    return fold_results

def generate_distance_data(n_samples, VOL_SIZE, swapped=False):
    X = torch.zeros((n_samples, 1, VOL_SIZE, VOL_SIZE, VOL_SIZE))
    Y = torch.randint(0, 2, (n_samples,)).float()
    
    for i in range(n_samples):
        label = Y[i]
        # Class 0: Close (4), Class 1: Far (12)
        dist = 4 if label == 0 else 12
        
        if not swapped:
            # Training distribution: Class 0 Left, Class 1 Right
            start_x = np.random.randint(2, 6) if label == 0 else np.random.randint(14, 18)
        else:
            # Testing distribution: Class 0 Right, Class 1 Left
            start_x = np.random.randint(14, 18) if label == 0 else np.random.randint(2, 6)
        y, z = np.random.randint(10, 20, size=2)
        # 3x3x3 cubes
        X[i, 0, start_x:start_x+3, y:y+3, z:z+3] = 1.0
        X[i, 0, start_x+dist:start_x+dist+3, y:y+3, z:z+3] = 1.0
        
    return X, Y

def plot_obj_detect(VOL_SIZE):
    # Generate samples
    X_train, Y_train = generate_distance_data(10, VOL_SIZE=VOL_SIZE, swapped=False)
    X_test, Y_test = generate_distance_data(10, VOL_SIZE=VOL_SIZE, swapped=True)

    # Find one of each class from train and test
    idx_t0 = (Y_train == 0).nonzero()[0].item()
    idx_t1 = (Y_train == 1).nonzero()[0].item()
    idx_s0 = (Y_test == 0).nonzero()[0].item()
    idx_s1 = (Y_test == 1).nonzero()[0].item()

    samples = [X_train[idx_t0], X_train[idx_t1], X_test[idx_s0], X_test[idx_s1]]
    titles = ["Class 0 (Close/Left)", "Class 1 (Far/Right)", 
              "Class 0 (Close/Right)", "Class 1 (Far/Left)"]

    fig, axes = plt.subplots(2, 4, figsize=(10, 5), facecolor='white')
    
    for i, (vol, title) in enumerate(zip(samples, titles)):
        # 1. SPATIAL VIEW: Use Max Projection instead of a single slice
        # This ensures we see the dots regardless of where they are in 3D
        spatial_proj = torch.max(vol[0], dim=1)[0].numpy()
        
        axes[0, i].imshow(spatial_proj, cmap='hot', vmin=0, vmax=1)
        axes[0, i].set_title(f"{title}", fontsize=12)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        # 2. FOURIER VIEW: 2D FFT of the projection
        f = torch.fft.fft2(torch.from_numpy(spatial_proj))
        f_shift = torch.fft.fftshift(torch.abs(f))
        # Use a small epsilon and log scale to see the fringes
        f_mag = torch.log(f_shift + 1e-3).numpy()
        
        axes[1, i].imshow(f_mag, cmap='viridis')
        #axes[1, i].set_title("Fourier tokenization", fontsize=20)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    plt.suptitle('Inference distribution shift',fontsize=20)
    axes[0,0].set_ylabel('Image \nspace',fontsize=20)
    axes[1,0].set_ylabel('Fourier \nspace',fontsize=20)
    
    # Add centered labels at the bottom for Train and Test groups
    fig.text(0.31, 0.02, 'Training distribution', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.71, 0.02, 'Testing distribution', ha='center', fontsize=14, fontweight='bold')

    # Use rect to prevent tight_layout from overlapping the suptitle and bottom labels
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def psnr(gt, recon):
    mse = np.mean((gt - recon)**2)
    return 20 * np.log10(1.0 / np.sqrt(mse))

def plot_sl_recon(scenarios, noise_lvl, X_train_pca, K, img_size):
    fig, axes = plt.subplots(3, 4, figsize=(14, 14))
    plt.subplots_adjust(wspace=0.05, hspace=0.3) 

    for i, (gt, row_name, labels) in enumerate(scenarios):
        effective_noise = 0.12 if i == 2 else noise_lvl
        noisy = gt + np.random.normal(0, effective_noise, gt.shape)
        
        _, r_pca = pca_tokenize(img_size, X_train_pca, noisy, K)
        _, r_fou = fourier_tokenize(img_size, noisy, K)
        _, r_lap = laplacian_tokenize(img_size, gt, noisy, K, topo_labels=labels) 
        
        recons = [r_pca, r_fou, r_lap]
        titles = ["PCA basis", "Fourier basis", "Laplacian basis"]
        psnrs = [psnr(gt, r) for r in recons]
        
        axes[i, 0].imshow(gt, cmap='bone', vmin=0, vmax=1)
        axes[i, 0].set_title("Ground Truth", fontsize=20)
        axes[i, 0].set_ylabel(row_name,fontsize=20)

        for j, r in enumerate(recons):
            is_winner = psnrs[j] == max(psnrs)
            axes[i, j+1].imshow(r, cmap='bone', vmin=0, vmax=1)
            axes[i, j+1].set_title(f"{titles[j]}\n{psnrs[j]:.1f} dB", 
                                fontsize=16, fontweight='bold' if is_winner else 'normal',
                                )

    for ax in axes.flatten():
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f'Spectral tokenization by basis', fontsize=20, y=0.95)
    plt.show()

import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_auc_score


def delong_roc_variance(y_true, predictions):
    """Computes the structural components (V10 and V01) underlying DeLong's
    variance estimator for the AUC of a single set of predictions."""
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    m = np.sum(y_true == 1)
    n = np.sum(y_true == 0)

    pos = predictions[y_true == 1]
    neg = predictions[y_true == 0]

    v10 = np.array([np.sum(neg < p) + 0.5 * np.sum(neg == p) for p in pos]) / n
    v01 = np.array([np.sum(pos > p) + 0.5 * np.sum(pos == p) for p in neg]) / m
    return v10, v01


def compute_delong_paired(y_true, preds_baseline, preds_model):
    """Paired DeLong test for ROC AUC comparison, using rank-based structural
    components and their covariance across two correlated ROC curves. Uses
    the survival function to retain precision for extreme p-values."""
    y_true = np.asarray(y_true)
    auc_base = roc_auc_score(y_true, preds_baseline)
    auc_mod = roc_auc_score(y_true, preds_model)

    v10_base, v01_base = delong_roc_variance(y_true, preds_baseline)
    v10_mod, v01_mod = delong_roc_variance(y_true, preds_model)

    m = np.sum(y_true == 1)
    n = np.sum(y_true == 0)

    S10 = np.cov(v10_base, v10_mod)
    S01 = np.cov(v01_base, v01_mod)

    var_diff = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + \
               (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n

    diff = auc_base - auc_mod
    z = diff / np.sqrt(max(var_diff, 1e-12))
    p_val = 2 * stats.norm.sf(abs(z))
    return z, p_val


def compute_mcnemar(y_true, preds_base_bin, preds_mod_bin):
    """McNemar's test with continuity correction, using the survival
    function to retain precision for extreme p-values."""
    y_true = np.asarray(y_true)
    b = np.sum((preds_base_bin == y_true) & (preds_mod_bin != y_true))
    c = np.sum((preds_base_bin != y_true) & (preds_mod_bin == y_true))
    if (b + c) == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = stats.chi2.sf(chi2, df=1)
    return chi2, p_val

def fmt_cv(key, cells):
    mean_val, std_val = cells[key]
    return f"{mean_val:.3f}±{std_val:.3f}"

def get_stratified_subset(dataset, per_class):
    indices = []
    labels = dataset.targets.numpy()
    for i in range(10):
        idx = np.where(labels == i)[0]
        selected = np.random.choice(idx, per_class, replace=False)
        indices.extend(selected)
    imgs = torch.stack([dataset[i][0] for i in indices])
    targets = torch.tensor([dataset[i][1] for i in indices])
    return imgs, targets

def smooth_curve(x, y, n):
    x, y = np.array(x), np.array(y)
    if len(x) <= 2: return x, y
    x_log = np.log10(x)
    x_new_log = np.linspace(x_log.min(), x_log.max(), n)
    interp = PchipInterpolator(x_log, y)
    y_smooth = interp(x_new_log)
    return 10**x_new_log, np.clip(y_smooth, 0, 1.0)

def compute_multiclass_metrics(probs, y_true):
    return {
        'T1': accuracy_score(y_true, np.argmax(probs, axis=1)),
        'T3': top_k_accuracy_score(y_true, probs, k=3, labels=np.arange(10)),
        'T5': top_k_accuracy_score(y_true, probs, k=5, labels=np.arange(10)),
        'AUC': roc_auc_score(y_true, probs, multi_class='ovr', labels=np.arange(10))
    }