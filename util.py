import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from collections import defaultdict, Counter
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import pandas as pd
import random
import tempfile
import gzip
import shutil
from loader import QSM_Dataset, fast_resample_sharp
import nibabel as nib
import os
import tempfile
import shutil
import gzip
import numpy as np
import pandas as pd
import nibabel as nib

def prepare_qsm_dataset(source_type, nii_path, seg_path, csv_path, cache_path, 
                        load_cache=False, limit=None, mask_crop_fn=None,
                        unlabeled_nii_path=None, ext_mask_dir=None, cv_pad=True):

    print(f"Preparing {source_type} dataset (cv_pad={cv_pad})")
    
    # --- RESOLUTION CORRECTION LOGIC ---
    resample_fn = None
    if source_type.upper() == 'CHH':
        from util import fast_resample_sharp # Assuming this import exists in your env
        resample_fn = fast_resample_sharp 

    # File preparation
    if source_type.upper() == 'CHH':
        final_nii_dir = tempfile.mkdtemp()
        final_seg_dir = tempfile.mkdtemp()
        
        # 1. Process standard CHH files
        all_files = [f for f in os.listdir(nii_path) if f.startswith('000') and f.endswith('.nii.gz')]
        for f in all_files:
            try:
                sub_id = int(f.split('_')[0])
                os.symlink(os.path.join(nii_path, f), os.path.join(final_nii_dir, f"qsm_{sub_id}.nii.gz"))
                roi_src = os.path.join(seg_path, f"{sub_id:02d}_roi_combined.nii")
                if os.path.exists(roi_src):
                    with open(roi_src, 'rb') as f_in, gzip.open(os.path.join(final_seg_dir, f"seg_{sub_id:02d}.nii.gz"), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except Exception: continue
        
        # --- CONSENSUS MASK FOR UNLABELED INJECTION ---
        template_mask_path = None
        existing_masks = [os.path.join(final_seg_dir, f) for f in os.listdir(final_seg_dir) if f.endswith('.nii.gz')]
        if existing_masks:
            try:
                print(f"Generating consensus mean mask from {len(existing_masks)} subjects...")
                mask_data_list = [nib.load(m).get_fdata() for m in existing_masks]
                ref_img = nib.load(existing_masks[0])
                mean_mask = (np.mean(mask_data_list, axis=0) > 0.5).astype(np.float32)
                template_mask_path = os.path.join(tempfile.gettempdir(), f"mean_mask_{source_type}.nii.gz")
                nib.save(nib.Nifti1Image(mean_mask, ref_img.affine, ref_img.header), template_mask_path)
            except Exception as e:
                print(f"Warning: Mean mask generation failed: {e}")

        # 2. Process additional unlabeled files (CHH Only)
        if unlabeled_nii_path and os.path.exists(unlabeled_nii_path):
            extra_files = [f for f in os.listdir(unlabeled_nii_path) if f.endswith('.nii.gz')]
            print(f"Injecting {len(extra_files)} unlabeled subjects for pretraining...")
            for i, f in enumerate(extra_files):
                try:
                    fake_id = 9000 + i
                    img_src = os.path.join(unlabeled_nii_path, f)
                    target_link = os.path.join(final_nii_dir, f"qsm_{fake_id}.nii.gz")
                    
                    if not os.path.exists(target_link):
                        os.symlink(img_src, target_link)
                        if template_mask_path:
                            target_mask = os.path.join(final_seg_dir, f"seg_{fake_id:02d}.nii.gz")
                            tpl_img = nib.load(template_mask_path)
                            unl_img = nib.load(img_src)
                            tpl_data = tpl_img.get_fdata()
                            
                            z_tpl, z_unl = tpl_data.shape[2], unl_img.shape[2]
                            if z_tpl > z_unl:
                                diff = z_tpl - z_unl
                                start = diff // 2
                                end = start + z_unl
                                cropped_mask = tpl_data[:, :, start:end]
                                new_mask_img = nib.Nifti1Image(cropped_mask, unl_img.affine, unl_img.header)
                                nib.save(new_mask_img, target_mask)
                            else:
                                shutil.copy(template_mask_path, target_mask)
                except Exception: continue
    else:
        # MSW Case
        final_nii_dir, final_seg_dir = nii_path, seg_path

    # --- PULL SUBJECTS ON DISK ---
    all_qsm_files = [f for f in os.listdir(final_nii_dir) if f.startswith('qsm_') and f.endswith('.nii.gz')]
    all_subject_ids = {str(int(f.split('_')[1].split('.')[0])) for f in all_qsm_files}

    # Clinical configuration
    clinical_dict, label_map = {}, {}
    clin_dim = 9 if cv_pad else 6 
    
    if source_type.upper() == 'CHH':
        df_full = pd.read_csv(csv_path, header=None)
        data_df = df_full.iloc[2:].copy() 
        for _, row in data_df.iterrows():
            try:
                if pd.isna(row[0]): continue
                sub_id = str(int(float(row[0])))
                age, sex = pd.to_numeric(row[1], errors='coerce'), pd.to_numeric(row[2], errors='coerce')
                dur, ledd = pd.to_numeric(row[3], errors='coerce'), pd.to_numeric(row[4], errors='coerce')
                off_pre, on_pre = pd.to_numeric(row[9], errors='coerce'), pd.to_numeric(row[10], errors='coerce')
                post_stim = pd.to_numeric(row[11], errors='coerce')

                if cv_pad:
                    # [Age, Sex, 0, 0, Dur, LEDD, 0, Off, On]
                    vec = [age, sex, 0, 0, dur, ledd, 0, off_pre, on_pre]
                else:
                    # [Age, Sex, Dur, LEDD, Off, On]
                    vec = [age, sex, dur, ledd, off_pre, on_pre]
                
                clinical_dict[sub_id] = np.nan_to_num(np.array(vec, dtype=np.float32))
                
                if not pd.isna(off_pre) and not pd.isna(post_stim) and off_pre > 0:
                    improvement = (off_pre - post_stim) / off_pre
                    label_map[sub_id] = 1 if improvement >= 0.30 else 0
                else:
                    label_map[sub_id] = -1 
            except: continue
    else:
        # MSW Case - FORCED ALIGNMENT LOGIC
        motor_df = pd.read_csv(csv_path, header=1)
        motor_df.columns = [str(c).strip().replace('\n', ' ') for c in motor_df.columns]
        id_col, pre_col, post_col = 'CORNELL ID', 'OFF (pre-dbs updrs)', 'OFF meds ON stim 6mo'
        
        for _, row in motor_df.iterrows():
            try:
                raw_id = row.get(id_col)
                if pd.isna(raw_id): continue
                sub_id = str(int(float(raw_id)))
                
                # Extract individual values to ensure order control
                age = pd.to_numeric(row.get('Age', 0), errors='coerce')
                sex = pd.to_numeric(row.get('Sex', 0), errors='coerce')
                eth = pd.to_numeric(row.get('Ethnicity', 0), errors='coerce')
                race = pd.to_numeric(row.get('Race', 0), errors='coerce')
                dur = pd.to_numeric(row.get('Disease Duration (year)', 0), errors='coerce')
                ledd = pd.to_numeric(row.get('pre op levadopa equivalent dose (mg)', 0), errors='coerce')
                med_stat = pd.to_numeric(row.get('Test medication status', 0), errors='coerce')
                off_v = pd.to_numeric(row.get('OFF (pre-dbs updrs)', 0), errors='coerce')
                on_v = pd.to_numeric(row.get('ON (pre-dbs updrs)', 0), errors='coerce')

                if cv_pad:
                    # MSW 9-dim: [Age, Sex, Eth, Race, Dur, LEDD, MedStat, Off, On]
                    # This aligns perfectly with CHH indices for Age, Sex, Dur, LEDD, Off, On
                    vec = [age, sex, eth, race, dur, ledd, med_stat, off_v, on_v]
                else:
                    # MSW 6-dim: [Age, Sex, Dur, LEDD, Off, On]
                    # Forced to match CHH exactly
                    vec = [age, sex, dur, ledd, off_v, on_v]

                clinical_dict[sub_id] = np.nan_to_num(np.array(vec, dtype=np.float32))
                
                pre, post = pd.to_numeric(row.get(pre_col), errors='coerce'), pd.to_numeric(row.get(post_col), errors='coerce')
                if not pd.isna(pre) and not pd.isna(post) and pre > 0:
                    improvement = (pre - post) / pre
                    label_map[sub_id] = 1 if improvement >= 0.30 else 0
                else:
                    label_map[sub_id] = -1
            except: continue

    # Catch-all for subjects missing from CSV
    for sid in all_subject_ids:
        if sid not in clinical_dict: clinical_dict[sid] = np.zeros(clin_dim, dtype=np.float32)
        if sid not in label_map: label_map[sid] = -1 

    # Initialize Dataset
    from util import QSM_Dataset # Assuming QSM_Dataset is imported
    dataset = QSM_Dataset(
        final_nii_dir, 
        final_seg_dir, 
        mask_crop_fn, 
        clinical_dict, 
        label_map,
        limit=limit, 
        cache_path=cache_path, 
        load_cache=load_cache, 
        return_index=True,
        ext_mask_dir=ext_mask_dir,
        resample_fn=resample_fn
    )
    dataset.clin_dim = clin_dim
    
    disk_labels = [label_map[sid] for sid in all_subject_ids]
    print(f"Finished {source_type}: {len(dataset)} slices.")
    print(f"Responders: {disk_labels.count(1)} | Nonresponders: {disk_labels.count(0)} | Unlabeled: {disk_labels.count(-1)}")
    return dataset

def mask_crop(data,mask,pad,viz=False):
    z_idx = ~(mask==0).all((0,1))
    x_idx = ~(mask==0).all((1,2))
    y_idx = ~(mask==0).all((0,2))
    pad_x = pad[0]
    pad_y = pad[1]
    pad_z = pad[2]
    if viz:
        print('Cropping:',x_idx,y_idx,z_idx)
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

class RandomMasking(object):
    def __init__(self, mask_size=8, num_masks=2, p=0.5):
        self.mask_size, self.num_masks, self.p = mask_size, num_masks, p
    def __call__(self, tensor):
        if torch.rand(1).item() > self.p: return tensor
        _, h, w = tensor.shape
        for _ in range(self.num_masks):
            y, x = torch.randint(0, h - self.mask_size, (1,)), torch.randint(0, w - self.mask_size, (1,))
            tensor[:, y:y+self.mask_size, x:x+self.mask_size] = 0
        return tensor
    
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
        # Add epsilon to avoid division by zero
        class_weights.append(total_count / (count + 1e-8))
    return class_weights

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Stratified training-validation split
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
    # Diffuse labels at timestep t using the standard DDPM formula,
    # while preserving the original label shape to avoid broadcasting issues.
    
    # Args:
    #     x_start: [B, 1] tensor of original labels
    #     t: [B] tensor of timesteps
    #     alphas: [T] tensor of alpha values
    #     noise: optional noise tensor of shape [B, 1]
    
    # Returns:
    #     Noisy labels [B, 1]
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
    betas=None,     
    alphas=None,    
    alphas_cumprod=None  
):
   
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

def get_m(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    return [acc, sens, spec, auc, tp, tn, fp, fn]

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

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def report_split_stats(train_subs, val_subs, df):
    train_df = df[df['Subject'].isin(train_subs)]
    val_df = df[df['Subject'].isin(val_subs)]
    
    print(f"\n--- Split Composition ---")
    print(f"Train: {len(train_df)} subs | Responders: {train_df['label'].sum()} ({train_df['label'].mean():.2%})")
    print(f"Val:   {len(val_df)} subs | Responders: {val_df['label'].sum()} ({val_df['label'].mean():.2%})")
    print(f"Mean Age (Train vs Val): {train_df['Age'].mean():.1f} vs {val_df['Age'].mean():.1f}")
    print(f"-------------------------\n")