import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from util import mean_std_str, seed_everything
from networks import SpectralViT, SpatialViT


import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from util import mean_std_str, seed_everything
from networks import SpectralViT, SpatialViT

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from util import mean_std_str, seed_everything
from networks import SpectralViT, SpatialViT

def SpectralViT_cv(X_flat, Y, pca_candidates, device, epochs=20, lr=1e-3):
    """EXACT replication of the standalone Spectral script logic."""
    
    # 1. To match the standalone script, we must reset the seed to 0 
    # RIGHT before we start the candidate loop.
    seed_everything(0)
    
    # 2. Replicate the 'leaked' BATCH_SIZE = len(ts_y).
    # In a 5-fold split of IXI (566 samples), the last fold size is 113.
    # We simulate this by taking the last fold size of a dummy split.
    kf_peek = KFold(n_splits=5, shuffle=True, random_state=0)
    for _, test_idx in kf_peek.split(X_flat):
        fixed_batch_size = len(test_idx) 

    results = {}
    print(f"\nStarting Selection over PCA components: {pca_candidates}")

    for n_comp in pca_candidates:
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        fold_aucs = []
        
        for train_idx, test_idx in kf.split(X_flat):
            torch.cuda.empty_cache()
            
            # Fit PCA
            pca_model = PCA(n_components=n_comp, whiten=True).fit(X_flat[train_idx])
            tr_pca = torch.from_numpy(pca_model.transform(X_flat[train_idx])).float()
            tr_y = torch.from_numpy(Y[train_idx]).float()
            ts_pca = torch.from_numpy(pca_model.transform(X_flat[test_idx])).float().to(device)
            ts_y_fold = torch.from_numpy(Y[test_idx]).float().to(device)
            
            # Use the FIXED batch size to match the original script's global variable
            loader = DataLoader(TensorDataset(tr_pca, tr_y), batch_size=fixed_batch_size, shuffle=True)
            
            # Initialize Model
            model = SpectralViT(
                n_inputs=n_comp, n_heads=2, embed_dim=16, n_layers=4,
                patch_size=1, use_rank_weights=True, learnable_rank_weights=True,
                use_input_proj=True, use_pos_embed=True, pooling='mean',
                use_layer_norm=True, use_sigmoid=False
            ).to(device)
            
            opt = optim.AdamW(model.parameters(), lr=lr)
            sched = OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(loader), epochs=epochs)
            crit = nn.BCEWithLogitsLoss()
            
            for _ in range(epochs):
                model.train()
                for b_pca, b_y in loader:
                    b_pca, b_y = b_pca.to(device), b_y.to(device)
                    opt.zero_grad()
                    loss = crit(model(b_pca), b_y)
                    loss.backward()
                    opt.step()
                    sched.step()
            
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(ts_pca)).cpu().numpy()
                fold_aucs.append(roc_auc_score(ts_y_fold.cpu(), probs))
        
        results[n_comp] = fold_aucs
        print(f"  n_components={n_comp}: AUC = {mean_std_str(fold_aucs)}")

    return max(results, key=lambda k: np.mean(results[k]))

def SpatialViT_cv(X, Y, patch_candidates, device, vol_size=96, epochs=20, lr=1e-3, physical_bs=2, effective_bs=8):
    """EXACT replication of the standalone Spatial script logic."""
    # Reset seed to 0 before starting the spatial grid
    seed_everything(0)
    
    accumulation_steps = effective_bs // physical_bs
    results = {}
    print(f"\nStarting Selection over Patch Sizes: {patch_candidates}")

    for p_size in patch_candidates:
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        fold_aucs = []
        
        for train_idx, test_idx in kf.split(X):
            torch.cuda.empty_cache()
            
            tr_vol = torch.from_numpy(X[train_idx]).unsqueeze(1).float()
            tr_y   = torch.from_numpy(Y[train_idx]).float()
            ts_vol = torch.from_numpy(X[test_idx]).unsqueeze(1).float().to(device)
            ts_y_fold = torch.from_numpy(Y[test_idx]).float().to(device)
            
            loader = DataLoader(TensorDataset(tr_vol, tr_y), batch_size=physical_bs, shuffle=True)
            
            model = SpatialViT(
                vol_size=vol_size, patch_size=p_size, embed_dim=128, n_heads=4, n_layers=2,
                dropout=0.2, is_2d=False, use_cls_token=True, use_layer_norm=True, use_sigmoid=False
            ).to(device)
            
            opt = optim.AdamW(model.parameters(), lr=lr)
            steps_per_epoch = len(loader) // accumulation_steps
            sched = OneCycleLR(opt, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
            crit = nn.BCEWithLogitsLoss()
            
            for _ in range(epochs):
                model.train()
                for i, (b_vol, b_y) in enumerate(loader):
                    b_vol, b_y = b_vol.to(device), b_y.to(device)
                    loss = crit(model(b_vol), b_y) / accumulation_steps
                    loss.backward()
                    if (i + 1) % accumulation_steps == 0:
                        opt.step()
                        sched.step()
                        opt.zero_grad()
            
            model.eval()
            with torch.no_grad():
                test_probs = []
                for chunk in torch.split(ts_vol, physical_bs):
                    test_probs.append(torch.sigmoid(model(chunk)))
                probs = torch.cat(test_probs).cpu().numpy()
                fold_aucs.append(roc_auc_score(ts_y_fold.cpu(), probs))
                
        results[p_size] = fold_aucs
        print(f"  patch_size={p_size}: AUC = {mean_std_str(fold_aucs)}")

    return max(results, key=lambda k: np.mean(results[k]))

def SpectralViT_cv_regression(X_flat, Y, pca_candidates, device, epochs=20, lr=1e-3):
    """
    Regression analog of SpectralViT_cv: picks the number of PCA components that
    gives the lowest held-out MAE (in years) instead of the highest AUC.

    Age is standardized (mean/std fit on the TRAIN fold only) before being handed to
    the network, since MSE/OneCycleLR are easier to optimize against a ~N(0,1) target
    than a raw ~[8, 100] year value; predictions are converted back to years before
    MAE is computed, so the reported numbers are directly interpretable.
    """
    seed_everything(0)
    kf_peek = KFold(n_splits=5, shuffle=True, random_state=0)
    for _, test_idx in kf_peek.split(X_flat):
        fixed_batch_size = len(test_idx)

    results = {}
    print(f"\nStarting Selection over PCA components: {pca_candidates}")

    for n_comp in pca_candidates:
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        fold_maes = []

        for train_idx, test_idx in kf.split(X_flat):
            torch.cuda.empty_cache()

            pca_model = PCA(n_components=n_comp, whiten=True).fit(X_flat[train_idx])
            tr_pca = torch.from_numpy(pca_model.transform(X_flat[train_idx])).float()

            y_mean, y_std = Y[train_idx].mean(), Y[train_idx].std() + 1e-8
            tr_y = torch.from_numpy((Y[train_idx] - y_mean) / y_std).float()
            ts_pca = torch.from_numpy(pca_model.transform(X_flat[test_idx])).float().to(device)
            ts_y_fold = Y[test_idx]  # kept in raw years for MAE

            loader = DataLoader(TensorDataset(tr_pca, tr_y), batch_size=fixed_batch_size, shuffle=True)

            model = SpectralViT(
                n_inputs=n_comp, n_heads=2, embed_dim=16, n_layers=4,
                patch_size=1, use_rank_weights=True, learnable_rank_weights=True,
                use_input_proj=True, use_pos_embed=True, pooling='mean',
                use_layer_norm=True, use_sigmoid=False
            ).to(device)

            opt = optim.AdamW(model.parameters(), lr=lr)
            sched = OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(loader), epochs=epochs)
            crit = nn.MSELoss()

            for _ in range(epochs):
                model.train()
                for b_pca, b_y in loader:
                    b_pca, b_y = b_pca.to(device), b_y.to(device)
                    opt.zero_grad()
                    loss = crit(model(b_pca), b_y)
                    loss.backward()
                    opt.step()
                    sched.step()

            model.eval()
            with torch.no_grad():
                preds = model(ts_pca).cpu().numpy() * y_std + y_mean
                fold_maes.append(mean_absolute_error(ts_y_fold, preds))

        results[n_comp] = fold_maes
        print(f"  n_components={n_comp}: MAE = {mean_std_str(fold_maes)} yrs")

    return min(results, key=lambda k: np.mean(results[k]))

def SpatialViT_cv_regression(X, Y, patch_candidates, device, vol_size=96, epochs=20, lr=1e-3,
                              physical_bs=2, effective_bs=8):
    """Regression analog of SpatialViT_cv: picks the patch size with lowest held-out MAE."""
    seed_everything(0)
    accumulation_steps = effective_bs // physical_bs
    results = {}
    print(f"\nStarting Selection over Patch Sizes: {patch_candidates}")

    for p_size in patch_candidates:
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        fold_maes = []

        for train_idx, test_idx in kf.split(X):
            torch.cuda.empty_cache()

            tr_vol = torch.from_numpy(X[train_idx]).unsqueeze(1).float()
            y_mean, y_std = Y[train_idx].mean(), Y[train_idx].std() + 1e-8
            tr_y = torch.from_numpy((Y[train_idx] - y_mean) / y_std).float()
            ts_vol = torch.from_numpy(X[test_idx]).unsqueeze(1).float().to(device)
            ts_y_fold = Y[test_idx]

            loader = DataLoader(TensorDataset(tr_vol, tr_y), batch_size=physical_bs, shuffle=True)

            model = SpatialViT(
                vol_size=vol_size, patch_size=p_size, embed_dim=128, n_heads=4, n_layers=2,
                dropout=0.2, is_2d=False, use_cls_token=True, use_layer_norm=True, use_sigmoid=False
            ).to(device)

            opt = optim.AdamW(model.parameters(), lr=lr)
            steps_per_epoch = len(loader) // accumulation_steps
            sched = OneCycleLR(opt, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
            crit = nn.MSELoss()

            for _ in range(epochs):
                model.train()
                for i, (b_vol, b_y) in enumerate(loader):
                    b_vol, b_y = b_vol.to(device), b_y.to(device)
                    loss = crit(model(b_vol), b_y) / accumulation_steps
                    loss.backward()
                    if (i + 1) % accumulation_steps == 0:
                        opt.step()
                        sched.step()
                        opt.zero_grad()

            model.eval()
            with torch.no_grad():
                test_preds = []
                for chunk in torch.split(ts_vol, physical_bs):
                    test_preds.append(model(chunk))
                preds = torch.cat(test_preds).cpu().numpy() * y_std + y_mean
                fold_maes.append(mean_absolute_error(ts_y_fold, preds))

        results[p_size] = fold_maes
        print(f"  patch_size={p_size}: MAE = {mean_std_str(fold_maes)} yrs")

    return min(results, key=lambda k: np.mean(results[k]))

def ClinicalTransformer_cv(model_class, model_kwargs, pos_weight_grid, outer_skf, unique_subjs, y_unique, tr_subj_map, X_tr_clin, y_tr_slices, NEG_WEIGHT, criterion, JITTER_STD, EPOCHS, device):
    """
    Optimizes pos_weight by re-initializing the model for every fold to prevent data leakage
    and cumulative training bias.
    """
    print("\n Optimizing `pos_weight` via clinical transformer...")
    weight_results = {k: [] for k in pos_weight_grid}
    
    for pos_weight in pos_weight_grid:
        fold_aucs = []
        for fold_idx, (train_subj_idx, val_subj_idx) in enumerate(outer_skf.split(unique_subjs, y_unique)):
            # --- FIX: Re-initialize the model for every fold ---
            model = model_class(**model_kwargs).to(device)
            
            train_subjects, val_subjects = unique_subjs[train_subj_idx], unique_subjs[val_subj_idx]
            train_mask, val_mask = np.isin(tr_subj_map, train_subjects), np.isin(tr_subj_map, val_subjects)

            X_train_clin_t = torch.tensor(X_tr_clin[train_mask], dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_tr_slices[train_mask], dtype=torch.float32).to(device)
            X_val_clin_t = torch.tensor(X_tr_clin[val_mask], dtype=torch.float32).to(device)
            y_val = y_tr_slices[val_mask]

            # Optimizer must be linked to the fresh model's parameters
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            w = torch.where(y_train_t == 0, torch.tensor(NEG_WEIGHT, device=device), torch.tensor(pos_weight, device=device))

            for epoch in range(EPOCHS):
                model.train()
                optimizer.zero_grad()
                X_jitter = X_train_clin_t + torch.randn_like(X_train_clin_t) * JITTER_STD
                loss = criterion(model(X_jitter), y_train_t, weight=w)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_probs = model(X_val_clin_t).cpu().numpy()
                val_subj_probs = [val_probs[tr_subj_map[val_mask] == s].mean() for s in val_subjects]
                val_subj_labels = [y_val[tr_subj_map[val_mask] == s][0] for s in val_subjects]
                fold_aucs.append(roc_auc_score(val_subj_labels, val_subj_probs))

        weight_results[pos_weight] = fold_aucs
        print(f"  Pos Weight={pos_weight}: AUC = {mean_std_str(fold_aucs)}")

    # Selection logic based on your original snippet (minimizing standard deviation)
    SELECTED_POS_WEIGHT = pos_weight_grid[np.argmin([np.std(weight_results[pw]) for pw in pos_weight_grid])]
    print(f"Selected positive weight: {SELECTED_POS_WEIGHT}")
    return SELECTED_POS_WEIGHT


def ResidualSpectralViT_cv(model_classes, model_kwargs, pca_components_grid, outer_skf, unique_subjs, y_unique, tr_subj_map, X_tr_slices, X_tr_clin, y_tr_slices, NEG_WEIGHT, SELECTED_POS_WEIGHT, criterion, JITTER_STD, EPOCHS, device):
    pca_results = {k: [] for k in pca_components_grid}
    print("\n Optimizing `n_components` with spectral ViT")
    
    # Extract classes from the passed dictionary for readability
    CT_Class = model_classes['clinical']
    ViT_Class = model_classes['vit']
    Wrapper_Class = model_classes['wrapper']

    for n_comp in pca_components_grid:
        fold_aucs = []
        for fold_idx, (train_subj_idx, val_subj_idx) in enumerate(outer_skf.split(unique_subjs, y_unique)):
            train_subjects, val_subjects = unique_subjs[train_subj_idx], unique_subjs[val_subj_idx]
            train_mask, val_mask = np.isin(tr_subj_map, train_subjects), np.isin(tr_subj_map, val_subjects)

            # Fit Scaler and PCA only on training data to prevent leakage
            f_img_scaler = StandardScaler().fit(X_tr_slices[train_mask])
            f_pca = PCA(n_components=n_comp, random_state=0, whiten=True).fit(
                f_img_scaler.transform(X_tr_slices[train_mask])
            )

            # Prepare Tensors
            X_tr_clin_t = torch.tensor(X_tr_clin[train_mask], dtype=torch.float32).to(device)
            X_tr_pca_t = torch.tensor(f_pca.transform(f_img_scaler.transform(X_tr_slices[train_mask])), dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_tr_slices[train_mask], dtype=torch.float32).to(device)

            X_val_clin_t = torch.tensor(X_tr_clin[val_mask], dtype=torch.float32).to(device)
            X_val_pca_t = torch.tensor(f_pca.transform(f_img_scaler.transform(X_tr_slices[val_mask])), dtype=torch.float32).to(device)
            y_val = y_tr_slices[val_mask]

            # Initialize Clinical Transformer
            m_ct = CT_Class(n_inputs=X_tr_clin.shape[1]).to(device)
            opt_ct = optim.Adam(m_ct.parameters(), lr=1e-4)
            w = torch.where(y_train_t == 0, torch.tensor(NEG_WEIGHT, device=device), torch.tensor(SELECTED_POS_WEIGHT, device=device))
            
            # Pre-train Clinical Transformer
            for _ in range(EPOCHS // 2):
                m_ct.train()
                opt_ct.zero_grad()
                loss = criterion(m_ct(X_tr_clin_t + torch.randn_like(X_tr_clin_t) * JITTER_STD), y_train_t, weight=w)
                loss.backward()
                opt_ct.step()
            
            m_ct.eval()
            for p in m_ct.parameters():
                p.requires_grad_(False)

            # Spectral ViT
            vit_params = model_kwargs['vit'].copy()
            vit_params['n_inputs'] = n_comp
            
            model = Wrapper_Class(
                m_ct, 
                ViT_Class(**vit_params)
            ).to(device)
            
            optimizer = optim.Adam(model.m_res.parameters(), lr=5e-4)

            # Train Combined Model
            for epoch in range(EPOCHS):
                model.train()
                optimizer.zero_grad()
                loss = criterion(model(X_tr_pca_t, X_tr_clin_t), y_train_t, weight=w)
                loss.backward()
                optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                val_probs = model(X_val_pca_t, X_val_clin_t).cpu().numpy()
                val_subj_probs = [val_probs[tr_subj_map[val_mask] == s].mean() for s in val_subjects]
                val_subj_labels = [y_val[tr_subj_map[val_mask] == s][0] for s in val_subjects]
                fold_aucs.append(roc_auc_score(val_subj_labels, val_subj_probs))

        pca_results[n_comp] = fold_aucs
        print(f"  PCA={n_comp}: AUC = {mean_std_str(fold_aucs)}")

    # Select best PCA components based on Mean AUC
    N_PCA_COMPONENTS = pca_components_grid[np.argmax([np.mean(pca_results[n]) for n in pca_components_grid])]
    print(f"Selected PCA components: {N_PCA_COMPONENTS}")
    return N_PCA_COMPONENTS