import torch
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import affine_transform, gaussian_filter

def fast_resample_sharp(volume, msw_res=0.5, chh_res=0.9, target_shape=(64, 64, 72), sharpen=True):
    m_res, c_res = float(msw_res), float(chh_res)
    t_shape = target_shape if target_shape else (64, 64, 72)
    
    scaling_factor = m_res / c_res  
    in_center = np.array(volume.shape) / 2.0
    out_center = np.array(t_shape) / 2.0
    offset = in_center - (scaling_factor * out_center)
    
    # 1. Resample with 'nearest' to fill the frame
    resampled = affine_transform(
        volume,
        matrix=scaling_factor * np.eye(3),
        offset=offset,
        output_shape=t_shape,
        order=3,
        mode='nearest' 
    )
    
    # 2. Apply sharpening GENTLY
    if sharpen:
        # We sharpen BEFORE masking to avoid creating a new edge
        blurred = gaussian_filter(resampled, sigma=0.8)
        resampled = resampled + 0.5 * (resampled - blurred)

    # 3. SOFT FEATHERING
    # This creates a mask that is 1.0 in the center and fades to 0.0 at edges
    # avoiding the "hard zero" that causes the zero-padding artifact.
    mask = np.ones(t_shape)
    # Use a large-sigma Gaussian to create a smooth falloff at the very edges
    mask = gaussian_filter(mask, sigma=1.5) 
    # Normalize mask so center is 1.0
    mask = mask / mask.max()

    return (resampled * mask).astype(np.float32)

class QSM_Dataset(Dataset):
    def __init__(self, nii_dir, seg_dir, mask_crop_fn, clinical_dict, label_map, 
                 limit=None, cache_path=None, load_cache=False, return_index=False,
                 slices_per_subject=None, random_seed=0, ext_mask_dir=None,
                 resample_fn=None):
        
        self.samples = []
        self.volumes = {} 
        self.seg_masks = {}
        self.clinical_dict = clinical_dict
        self.label_map = label_map
        self.train_mode = False 
        self.return_index = return_index
        self.slices_per_subject = slices_per_subject
        self.random_seed = random_seed
        self.ext_mask_dir = ext_mask_dir
        self.resample_fn = resample_fn

        if load_cache and cache_path and os.path.exists(cache_path):
            cached_data = torch.load(cache_path)
            samples = cached_data['samples']
            mismatches = [s for s in samples if s['label'] != self.label_map.get(str(s['sub_id']), -1)]
            
            if len(mismatches) > 0:
                print(f"Updating {len(mismatches)} labels in cache...")
                for s in samples:
                    s['label'] = self.label_map.get(str(s['sub_id']), -1)
                cached_data['samples'] = samples
                torch.save(cached_data, cache_path)

            self.volumes = cached_data['volumes']
            self.samples = samples
            self.seg_masks = cached_data.get('seg_masks', {})
            print(f"Loaded cache with {len(self.samples)} slices.")
            return

        all_potential = [f for f in os.listdir(nii_dir) if f.startswith('qsm_') and f.endswith('.nii.gz')]
        loaded_count = 0
        
        for f in tqdm(all_potential, desc="Caching Volumes"):
            if limit and loaded_count >= limit: break
            try:
                raw_id = f.split('_')[1].split('.')[0]
                sub_id_str = str(int(raw_id))
                sub_id_int = int(sub_id_str)
                case_id = f"{sub_id_int:02d}"
                
                mask_path = os.path.join(seg_dir, f'seg_{case_id}.nii.gz')
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(seg_dir, f'seg_{case_id}.nii')
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(seg_dir, f'{sub_id_int}_roi_combined.nii')
                
                if not os.path.exists(mask_path): continue
                
                actual_label = self.label_map.get(sub_id_str, -1)
                if actual_label == -1 and sub_id_str not in self.clinical_dict: continue 

                # 3. Load & Force Alignment
                img_nib = nib.load(os.path.join(nii_dir, f))
                mask_nib = nib.load(mask_path)

                # Header Hijack: Force mask into QSM physical space
                if img_nib.shape == mask_nib.shape:
                    mask_nib = nib.Nifti1Image(mask_nib.get_fdata(), img_nib.affine, img_nib.header)

                img_obj = nib.as_closest_canonical(img_nib)
                mask_obj = nib.as_closest_canonical(mask_nib)
                
                raw_data = img_obj.get_fdata()
                mask_data = mask_obj.get_fdata()

                # 4. Native Anchor Calculation (Disambiguate CHH vs MSW)
                unique_labels = set(np.unique(mask_data))

                # Logic: Prioritize the "Uniquely Identifying" labels
                if 3 in unique_labels or 4 in unique_labels:
                    # Cohort: CHH. Target: 1-4. Ignore: 5-6 (Dentate)
                    anchor_labels = [1, 2, 3, 4]
                elif 7 in unique_labels or 8 in unique_labels:
                    # Cohort: MSW. Target: 5-8. Ignore: 1-2 (Other)
                    anchor_labels = [5, 6, 7, 8]
                else:
                    anchor_labels = [l for l in unique_labels if l > 0]

                # CRITICAL: Strip out non-target labels (kills Dentate/Red Nucleus)
                mask_data = np.where(np.isin(mask_data, anchor_labels), mask_data, 0)
                anchor_mask = (mask_data > 0).astype(np.float32)

                if np.sum(anchor_mask) == 0:
                    print(f"Skipping {sub_id_int}: Target ROI not found after filtering.")
                    continue

                # 5. Conditional Crop & Resample
                target_shape = (64, 64, 72) # Explicitly define here to avoid scoping issues

                if self.resample_fn is not None:
                    # CHH MODE
                    # Safety check on resolutions to prevent the NoneType error
                    m_res, c_res = 0.5, 0.9 
                    
                    # Calculate crop size in native space
                    native_crop_shape = tuple(int(round((ts * m_res) / c_res)) for ts in target_shape)
                    
                    # Crop first
                    raw_patch = mask_crop_fn(raw_data, anchor_mask, native_crop_shape)
                    mask_patch = mask_crop_fn(mask_data, anchor_mask, native_crop_shape)
                    
                    # Resample to standardized 0.5mm 64x64x72
                    standardized_data = self.resample_fn(raw_patch, msw_res=m_res, chh_res=c_res, 
                                                        target_shape=target_shape, sharpen=True)
                    m_patch = self.resample_fn(mask_patch, msw_res=m_res, chh_res=c_res, 
                                            target_shape=target_shape, sharpen=False)
                    m_patch = (m_patch > 0.5).astype(np.uint8)
                else:
                    # MSW MODE
                    standardized_data = mask_crop_fn(raw_data, anchor_mask, target_shape)
                    m_patch = (mask_crop_fn(mask_data, anchor_mask, target_shape) > 0).astype(np.uint8)

                # --- SANITY CHECK PLOT ---
                if loaded_count == 0:
                    print(f"DEBUG: Showing orientation for Subject {sub_id_int}...")
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    mid_z = standardized_data.shape[2] // 2
                    ax[0].imshow(np.rot90(standardized_data[:, :, mid_z]), cmap='gray')
                    ax[0].set_title(f"QSM Crop (Sub {sub_id_int})")
                    ax[1].imshow(np.rot90(m_patch[:, :, mid_z]), cmap='jet')
                    ax[1].set_title("Target ROI (Filtered)")
                    plt.show()

                # 6. Standardization
                standardized_data[standardized_data <= -2000] = 0.0
                if self.ext_mask_dir: 
                    skull_mask_path = os.path.join(self.ext_mask_dir, f"mag_{case_id}BrainExtractionMask.nii.gz")
                    if os.path.exists(skull_mask_path):
                        skull_mask = mask_crop_fn(nib.load(skull_mask_path).get_fdata(), anchor_mask, target_shape)
                        standardized_data = standardized_data * (skull_mask > 0)

                standardized_data = np.clip(standardized_data, -250.0, 250.0) / 250.0

                # 7. Axial alignment
                processed_vol = standardized_data.astype(np.float32) 
                
                # 8. Store
                self.volumes[sub_id_int] = processed_vol
                self.seg_masks[sub_id_int] = m_patch

                total_slices = processed_vol.shape[2] 
                rng = np.random.RandomState(self.random_seed + sub_id_int)
                rng = np.random.RandomState(self.random_seed + sub_id_int)
                
                if self.slices_per_subject is not None:
                    n_slices = self.slices_per_subject.get(actual_label, total_slices) if isinstance(self.slices_per_subject, dict) else self.slices_per_subject
                    slice_indices = rng.choice(total_slices, min(n_slices, total_slices), replace=False)
                else:
                    slice_indices = range(total_slices)
                
                for idx in slice_indices:
                    self.samples.append({'sub_id': sub_id_int, 'slice_idx': idx, 'label': actual_label})
                
                loaded_count += 1
            except Exception as e:
                print(f"Skipping {f}: {e}")

        if cache_path:
            torch.save({'volumes': self.volumes, 'samples': self.samples, 'seg_masks': self.seg_masks}, cache_path)
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, index):
            meta = self.samples[index]
            vol = self.volumes[meta['sub_id']]
            # This MUST pull the same plane as the sanity check plot
            img_slice = vol[:, :, meta['slice_idx']]
            img_tensor = torch.from_numpy(img_slice).unsqueeze(0) 
            
            clin_data = self.clinical_dict.get(str(meta['sub_id']), np.zeros(9))
            return img_tensor, torch.tensor(clin_data, dtype=torch.float32), int(meta['label']), index
    


class QSM_c1_Dataset(Dataset):
    def __init__(self, nii_dir, seg_dir, mask_crop_fn, clinical_dict, label_map, 
                 limit=None, cache_path=None, load_cache=False, return_index=False,
                 slices_per_subject=None, random_seed=0):
        """
        Args:
            slices_per_subject: None (use all), int (uniform), or dict {label: n_slices}
                Example: {0: 560, 1: 32, -1: 72} to replicate good cache balance
            random_seed: Seed for reproducible slice sampling
        """
        self.samples = []
        self.volumes = {} 
        self.seg_masks = {}
        self.clinical_dict = clinical_dict
        self.label_map = label_map
        self.train_mode = False 
        self.return_index = return_index
        self.slices_per_subject = slices_per_subject
        self.random_seed = random_seed

        # --- IMPROVED INTEGRITY CHECK ---
        if load_cache and cache_path and os.path.exists(cache_path):
            cached_data = torch.load(cache_path)
            samples = cached_data['samples']
            
            # Check if any subject in the cache is -1 BUT exists in our current label_map
            bad_samples = [s for s in samples if s['label'] == -1 and str(s['sub_id']) in self.label_map]
            
            if len(bad_samples) > 0:
                print(f"Found {len(bad_samples)} mismatched labels. Forcing a RE-CACHE...")
            else:
                self.volumes = cached_data['volumes']
                self.samples = samples
                self.seg_masks = cached_data.get('seg_masks', {})
                print(f"Loaded cache with {len(self.samples)} slices from {len(self.volumes)} subjects")
                return

        all_potential = [f for f in os.listdir(nii_dir) if f.startswith('qsm_') and f.endswith('.nii.gz')]
        loaded_count = 0
        
        for f in tqdm(all_potential, desc="Caching Volumes"):
            if limit and loaded_count >= limit: 
                break
            try:
                # 1. Standardize ID extraction
                raw_id = f.split('_')[1].split('.')[0]  # Get '101' from 'qsm_101.nii.gz'
                sub_id_str = str(int(raw_id))           # Remove leading zeros, then stringify
                sub_id_int = int(sub_id_str)            # Keep an int for volume storage
                
                case_id = f"{sub_id_int:02d}"
                mask_path = os.path.join(seg_dir, f'seg_{case_id}.nii.gz')
                if not os.path.exists(mask_path): 
                    continue
                
                # 2. Get Label (Force string keys to match clinical_dict/label_map)
                actual_label = self.label_map.get(sub_id_str, -1)
                
                # IF THE SUBJECT IS UNLABELED AND NOT IN CLINICAL DICT, SKIP
                if actual_label == -1 and sub_id_str not in self.clinical_dict:
                    continue 

                # 3. Preprocessing (Standardization)
                raw_data = nib.load(os.path.join(nii_dir, f)).get_fdata()
                mask_data = nib.load(mask_path).get_fdata()
                mask_data[mask_data <= 2] = 0
                
                img = mask_crop_fn(raw_data, mask_data, (72, 64, 64)) / 1000.0
                binary_mask = (mask_data > 0).astype(np.uint8)
                m_patch = mask_crop_fn(binary_mask, mask_data, (72, 64, 64))
                
                brain_indices = m_patch > 0
                if np.any(brain_indices):
                    # Local Z-score normalization based ONLY on brain tissue
                    img = (img - np.mean(img[brain_indices])) / (np.std(img[brain_indices]) + 1e-8)
                
                processed_vol = np.transpose(np.clip(img, -5.0, 5.0), (1, 2, 0)).astype(np.float32)
                
                # 4. Storage
                self.volumes[sub_id_int] = processed_vol
                self.seg_masks[sub_id_int] = m_patch
                
                # 5. Slice sampling - REPLICATE GOOD CACHE BALANCE
                total_slices = processed_vol.shape[2]  # Should be 72
                
                if self.slices_per_subject is not None:
                    if isinstance(self.slices_per_subject, dict):
                        # Per-class control
                        n_slices = self.slices_per_subject.get(actual_label, total_slices)
                    else:
                        # Uniform control
                        n_slices = self.slices_per_subject
                    
                    n_slices = min(n_slices, total_slices)
                    
                    # Reproducible sampling per subject
                    rng = np.random.RandomState(self.random_seed + sub_id_int)
                    slice_indices = rng.choice(total_slices, n_slices, replace=False)
                else:
                    # Use all slices
                    slice_indices = range(total_slices)
                
                for slice_idx in slice_indices:
                    self.samples.append({
                        'sub_id': sub_id_int, 
                        'slice_idx': slice_idx, 
                        'label': actual_label
                    })
                
                loaded_count += 1
            except Exception as e:
                print(f"Skipping {f}: {e}")
                continue
        
        # Final Check before saving
        final_labels = np.unique([s['label'] for s in self.samples])
        label_counts = {lbl: sum(1 for s in self.samples if s['label'] == lbl) for lbl in final_labels}
        print(f"\n>>> Cache built with {len(self.volumes)} subjects, {len(self.samples)} total slices")
        print(f">>> Label distribution:")
        for lbl in sorted(label_counts.keys()):
            name = "Responder" if lbl == 1 else "Non-Responder" if lbl == 0 else "Unlabeled"
            print(f"    Label {lbl} ({name}): {label_counts[lbl]} slices")
        
        if cache_path:
            torch.save({
                'volumes': self.volumes, 
                'samples': self.samples,
                'seg_masks': self.seg_masks
            }, cache_path)
            print(f">>> Cache saved to {cache_path}")
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, index):
        meta = self.samples[index]
        vol = self.volumes[meta['sub_id']]
        img = vol[:, :, meta['slice_idx']]
        img_tensor = torch.from_numpy(img).unsqueeze(0) / 5.0
        
        clin_data = self.clinical_dict.get(str(meta['sub_id']))
        if clin_data is not None:
            clin_vec = torch.tensor(clin_data, dtype=torch.float32)
        else:
            dim = getattr(self, 'clin_dim', 9) 
            clin_vec = torch.zeros(dim, dtype=torch.float32)
            
        label = int(meta['label'])
        
        # ALWAYS return 4 values to satisfy the (imgs, clin, _, _) loop
        return img_tensor, clin_vec, label, index