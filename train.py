import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

def get_mixed_data(dataset, NEG_OVERSAMPLE_FACTOR=1, factor=1, aug=None):
    l_imgs, l_clins, l_lbls, u_imgs, u_clins = [], [], [], [], []
    for idx in range(len(dataset)):
        img, clin, lbl, _ = dataset[idx]
        img_np = img.squeeze().numpy().flatten()
        if lbl != -1:
            reps = NEG_OVERSAMPLE_FACTOR if lbl == 0 else 1
            for _ in range(reps):
                l_imgs.append(img_np); l_clins.append(clin.numpy()); l_lbls.append(lbl)
                if aug and factor > 1:
                    for _ in range(factor-1):
                        a_img = aug(img.clone()).squeeze().numpy().flatten()
                        l_imgs.append(a_img); l_clins.append(clin.numpy()); l_lbls.append(lbl)
        else:
            u_imgs.append(img_np); u_clins.append(clin.numpy())
    return np.array(l_imgs), np.array(l_clins), np.array(l_lbls), np.array(u_imgs), np.array(u_clins)

def process_features(imgs, clins, pca, N_PCA_COMPONENTS, img_scaler):
    if imgs.size == 0: return np.zeros((0, N_PCA_COMPONENTS + clins.shape[1]))
    return np.hstack([pca.transform(img_scaler.transform(imgs)), clins])

def calibrate_balanced(model, attn, X_v, y_v,device):
    model.eval()
    if attn: attn.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_v, dtype=torch.float32).to(device)
        p = model(attn(x_t) if attn else x_t, mode='fine_tune').cpu().numpy()
    
    best_t, best_bal = 0.5, 0
    for t in np.linspace(0.01, 0.9, 100):
        score = balanced_accuracy_score(y_v, (p >= t))
        if score > best_bal:
            best_bal, best_t = score, t
    return best_t

def check_alignment(ds1, ds2):
    labels = ['Age', 'Sex', 'Dur', 'LEDD', 'Off-Pre', 'On-Pre']
    
    # Filter: Only take vectors that are NOT all zeros
    v1 = np.array([ds1.clinical_dict[sid] for sid in ds1.clinical_dict.keys() 
                   if not np.all(ds1.clinical_dict[sid] == 0)])
    v2 = np.array([ds2.clinical_dict[sid] for sid in ds2.clinical_dict.keys() 
                   if not np.all(ds2.clinical_dict[sid] == 0)]) 
    
    print(f"\n--- Verified Clinical Alignment (Excluding Zero-Filled Unlabeled) ---")
    print(f"{'Variable':<10} | {'MSW Mean':>10} | {'CHH Mean':>10} | {'Status'}")
    print("-" * 55)
    for i in range(6):
        m1, m2 = np.mean(v1[:, i]), np.mean(v2[:, i])
        status = "✅ OK" if abs(m1 - m2) < 15 else "❌ MISALIGNED"
        if i == 1: status = "✅ OK" if abs(m1 - m2) < 0.3 else "❌ MISALIGNED"
        print(f"{labels[i]:<10} | {m1:>10.2f} | {m2:>10.2f} | {status}")

def get_anatomy_only(imgs,pca,img_scaler):
    return pca.transform(img_scaler.transform(imgs))