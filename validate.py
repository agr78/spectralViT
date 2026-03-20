from util import calc_metrics, q_sample
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def calibrate(model, X_v, y_v, device):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(X_v, dtype=torch.float32).to(device), mode='fine_tune').cpu().numpy()
    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.1, 0.9, 81):
        score = f1_score(y_v, (p >= t))
        if score > best_f1: best_f1, best_t = score, t
    return best_t

def calibrate_attn(model, attn, X_v, y_v, device):
    model.eval(); attn.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_v, dtype=torch.float32).to(device)
        p = model(attn(x_t), mode='fine_tune').cpu().numpy()
    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.1, 0.9, 81):
        score = f1_score(y_v, (p >= t))
        if score > best_f1: best_f1, best_t = score, t
    return best_t

def calibrate_balanced(model, attn, X_v, y_v, device):
    model.eval()
    if attn: attn.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_v, dtype=torch.float32).to(device)
        p = model(attn(x_t) if attn else x_t, mode='fine_tune').cpu().numpy()
    best_t, best_bal = 0.5, 0
    for t in np.linspace(0.01, 0.9, 100): # Wider sweep
        score = balanced_accuracy_score(y_v, (p >= t))
        if score > best_bal:
            best_bal, best_t = score, t
    return best_t

def calibrate_balanced_full_attn(model, spec_attn, X_v, y_v, device):
    model.eval(); spec_attn.eval()
    with torch.no_grad():
        x_v_t = torch.tensor(X_v, dtype=torch.float32).to(device)
        p = model(spec_attn(x_v_t), mode='fine_tune').cpu().numpy()
    best_t, best_score = 0.5, 0
    for t in np.linspace(0.1, 0.9, 81):
        score = balanced_accuracy_score(y_v, (p >= t))
        if score > best_score: best_score, best_t = score, t
    return best_t

@torch.no_grad()
def val_model_stable(val_loader, device, model, loss_fn, val_subset, threshold=0.5):
    model.eval()
    val_loss = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for val_x, val_clin, val_y in val_loader:
            val_x, val_clin, val_y = val_x.to(device), val_clin.to(device), val_y.to(device)

            t_dummy = torch.zeros(val_x.size(0), device=device).long()

            # Model Inference
            if hasattr(model, 'clinical_mlp'): 
                if hasattr(model, 'time_mlp'): 
                    _, logits = model(val_x, t_dummy, clinical_vec=val_clin)
                else: 
                    logits, _ = model(val_x, clinical_vec=val_clin)
            else: 
                logits, _ = model(val_x, clinical_vec=val_clin)

            loss = loss_fn(logits, val_y)
            val_loss += loss.item()

            # ML conventions
            probs = torch.softmax(logits, dim=1)[:, 0]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(val_y.cpu().numpy())

    avg_loss = val_loss / len(val_loader)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Label Logic: 
    # all_labels is 0 for Non-Responder, 1 for Responder.
    # We want to treat Non-Responders (0) as the 'positive' case for metrics.
    metric_labels = (all_labels == 0).astype(int) 

    from util import calc_metrics
    from sklearn.metrics import roc_auc_score
    
    # Calculate AUC and other metrics specifically for the Non-Responder class
    metrics = calc_metrics(metric_labels, all_probs, threshold=threshold)
    auc = roc_auc_score(metric_labels, all_probs)
    
    # Return: Loss, Acc, Prec, Sens, Spec, AUC
    # (Double check your calc_metrics return order matches: Acc, Prec, Sens, Spec)
    return np.array([avg_loss, metrics[0], metrics[1], metrics[2], metrics[3], auc])

def val_model(val_loader, device, model, loss_fn, val_subset):
    """
    Universal validation routine supporting:
      - JointDiffusionUNet (expects x, t, cond)
      - JointDiffusionEncoderClassifier / Vanilla_Classifier (expects x only)
      - torchvision models (ResNet, ViT, etc., returns logits only)
    """
    y_true = []
    y_pred_diff = []
    val_loss_cls_diff = 0.0
    model.eval()

    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            t_dummy = torch.zeros(val_x.size(0), dtype=torch.long, device=device)

            # unwrap model if DataParallel is used
            model_unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model

            try:
                # Diffusion-style model with label embeddings
                cond_val = model_unwrapped.get_label_embedding(val_y)
                _, logits_val_diff = model(val_x, t_dummy, cond_val)

            except AttributeError:
                # Standard classifier (ResNet, encoder classifier, etc.)
                outputs = model(val_x)
                # handle possible tuple outputs
                if isinstance(outputs, tuple):
                    logits_val_diff = outputs[0]
                else:
                    logits_val_diff = outputs

            preds_diff = logits_val_diff.argmax(dim=1)
            val_loss_cls_diff += loss_fn(logits_val_diff, val_y).item() * val_x.size(0)

            y_true.extend(val_y.cpu().numpy())
            y_pred_diff.extend(preds_diff.cpu().numpy())

    # compute aggregated metrics
    val_loss_cls_diff /= len(val_subset)
    acc, prec, sens, spec = calc_metrics(y_true, y_pred_diff)
    try:
        auc = roc_auc_score(y_true, y_pred_diff)
    except Exception:
        auc = 0.0  # handle single-class edge cases safely

    return val_loss_cls_diff, acc, prec, sens, spec, auc

# -----------------------------
# Validation for scalar regression (any model)
# -----------------------------
@torch.no_grad()
def val_model_regression(loader, model, device):
    model.eval()
    mse_list, mae_list = [], []
    y_true_list, y_pred_list = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.float().to(device).unsqueeze(1)

        pred = model(x).squeeze(1)

        mse_list.append(F.mse_loss(pred, y.squeeze(1)).item())
        mae_list.append(torch.mean(torch.abs(pred - y.squeeze(1))).item())

        y_true_list.append(y.cpu())
        y_pred_list.append(pred.detach().cpu())

    y_true_all = torch.cat(y_true_list).float()
    y_pred_all = torch.cat(y_pred_list)
    ss_res = torch.sum((y_true_all - y_pred_all)**2)
    ss_tot = torch.sum((y_true_all - y_true_all.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return np.array([
        np.mean(mse_list),
        np.mean(mae_list),
        r2.item()
    ])

# -----------------------------------------------------
# Diffusion + regression validation
# -----------------------------------------------------
@torch.no_grad()
def val_model_regression_diffusion(loader, model, device, timesteps, alphas):
    model.eval()
    mse_list, mae_list = [], []
    y_true_list, y_pred_list = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.float().to(device).unsqueeze(1)

        # Sample random timestep and noise for diffusion input
        t = torch.randint(0, timesteps, (x.size(0),), device=device)
        noise = torch.randn_like(x)
        x_noisy = q_sample(x, t, alphas, noise)

        # Forward
        noise_pred, reg_pred = model(x_noisy, t)

        mse_list.append(F.mse_loss(reg_pred, y).item())
        mae_list.append(torch.mean(torch.abs(reg_pred - y)).item())

        y_true_list.append(y.detach().cpu())
        y_pred_list.append(reg_pred.detach().cpu())

    # Compute R^2
    y_true_all = torch.cat(y_true_list)
    y_pred_all = torch.cat(y_pred_list)
    ss_res = torch.sum((y_true_all - y_pred_all) ** 2)
    ss_tot = torch.sum((y_true_all - y_true_all.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return np.array([np.mean(mse_list), np.mean(mae_list), r2.item()])

class ValLoaderWrapper:
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset
    def __iter__(self):
        for batch in self.loader:
            yield batch[:3]
    def __len__(self):
        return len(self.loader)
