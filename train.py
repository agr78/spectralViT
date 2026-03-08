import numpy as np

def get_mixed_data(dataset, factor=1, aug=None):
    l_imgs, l_clins, l_lbls, u_imgs, u_clins = [], [], [], [], []
    NON_RES_TOTAL_AUG = 50 

    for idx in range(len(dataset)):
        img, clin, lbl, _ = dataset[idx]
        img_np = img.squeeze().numpy().flatten()
        if lbl != -1:
            l_imgs.append(img_np); l_clins.append(clin.numpy()); l_lbls.append(lbl)
            current_aug_limit = NON_RES_TOTAL_AUG if lbl == 0 else (factor - 1)
            if aug and current_aug_limit > 0:
                for _ in range(current_aug_limit):
                    a_img = aug(img.clone()).squeeze().numpy().flatten()
                    l_imgs.append(a_img); l_clins.append(clin.numpy()); l_lbls.append(lbl)
        else:
            u_imgs.append(img_np); u_clins.append(clin.numpy())
    return np.array(l_imgs), np.array(l_clins), np.array(l_lbls), np.array(u_imgs), np.array(u_clins)

def process_features(imgs, clins, pca, N_PCA_COMPONENTS, img_scaler):
    if imgs.size == 0: return np.zeros((0, N_PCA_COMPONENTS + clins.shape[1]))
    return np.hstack([pca.transform(img_scaler.transform(imgs)), clins])