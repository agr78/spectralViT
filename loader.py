import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset, Dataset, WeightedRandomSampler
import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

class QSM_c1_Dataset(Dataset):
    def __init__(self, nii_dir, seg_dir, mask_crop_fn, clinical_dict, label_map, limit=None, cache_path=None, load_cache=False):
        self.samples = []
        self.volumes = {} 
        self.clinical_dict = clinical_dict
        self.label_map = label_map
        self.transform = None   
        self.train_mode = False 
        
        if load_cache and cache_path and os.path.exists(cache_path):
            print(f">>> Loading preprocessed data from cache: {cache_path}...")
            cached_data = torch.load(cache_path,weights_only=False)
            self.volumes = cached_data['volumes']
            self.samples = cached_data['samples']
            return 

        all_potential = [f for f in os.listdir(nii_dir) if f.startswith('qsm_') and f.endswith('.nii.gz')]
        loaded_count = 0
        for f in tqdm(all_potential, desc="Caching Volumes"):
            if limit and loaded_count >= limit: break
            try:
                sub_id = int(f.split('_')[1])
                case_id = f"{sub_id:02d}"
                mask_path = os.path.join(seg_dir, f'seg_{case_id}.nii.gz')
                if not os.path.exists(mask_path): continue
                
                raw_data = nib.load(os.path.join(nii_dir, f)).get_fdata()
                mask_data = nib.load(mask_path).get_fdata()
                mask_data[mask_data <= 2] = 0
                binary_mask = (mask_data > 0).astype(np.uint8)
                
                img = mask_crop_fn(raw_data, mask_data, (72, 64, 64)) / 1000.0
                m_patch = mask_crop_fn(binary_mask, mask_data, (72, 64, 64))
                
                if img.shape != (72, 64, 64): continue
                brain_indices = m_patch > 0
                if np.any(brain_indices):
                    img = (img - np.mean(img[brain_indices])) / (np.std(img[brain_indices]) + 1e-8)
                
                processed_vol = np.transpose(np.clip(img, -5.0, 5.0), (1, 2, 0)).astype(np.float32)
                self.volumes[sub_id] = processed_vol
                
                actual_label = self.label_map.get(sub_id, -1)
                for slice_idx in range(72):
                    self.samples.append({'sub_id': sub_id, 'slice_idx': slice_idx, 'label': actual_label})
                loaded_count += 1
            except Exception: continue
        
        if cache_path:
            torch.save({'volumes': self.volumes, 'samples': self.samples}, cache_path)

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, index):
        meta = self.samples[index]
        vol = self.volumes[meta['sub_id']]
        img = vol[:, :, meta['slice_idx']]
        img_tensor = torch.from_numpy(img).unsqueeze(0) 
        if self.train_mode and self.transform:
            img_tensor = self.transform(img_tensor)
        img_tensor = img_tensor / 5.0 
        clin_data = self.clinical_dict.get(str(meta['sub_id']))
        clin_vec = torch.tensor(clin_data, dtype=torch.float32) if clin_data is not None else torch.zeros(getattr(self, 'clin_dim', 11))
        return img_tensor, clin_vec, int(meta['label']), index