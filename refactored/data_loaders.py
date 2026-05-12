import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, affine_transform
from matplotlib import pyplot as plt

def load_ixi_data(mni_dir, csv_path, vol_size=96):
    """
    Load IXI dataset for sex classification.
    
    Args:
        mni_dir: Directory containing MNI-registered NIfTI files
        csv_path: Path to CSV with subject metadata
        vol_size: Size of cropped volume (cubic)
        
    Returns:
        X: Array of volumes (N, vol_size, vol_size, vol_size)
        Y: Array of labels (N,) - 0 for male, 1 for female
    """
    df = pd.read_csv(csv_path)
    id_col = [c for c in df.columns if 'ID' in c.upper()][0]
    sex_col = [c for c in df.columns if 'SEX' in c.upper()][0]
    sex_lookup = dict(zip(df[id_col].astype(int), df[sex_col].map({1: 0, 2: 1})))
    
    files = sorted([f for f in os.listdir(mni_dir) if f.endswith('.nii.gz')])
    vols, labels = [], []
    
    for f in tqdm(files, desc="Loading Data"):
        match = re.search(r'(\d+)', f)
        if match and int(match.group(1)) in sex_lookup:
            img = nib.load(os.path.join(mni_dir, f)).get_fdata()
            c = np.array(img.shape) // 2
            r = vol_size // 2
            crop = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r, c[2]-r:c[2]+r]
            if crop.shape == (vol_size, vol_size, vol_size):
                crop = (crop - np.mean(crop)) / (np.std(crop) + 1e-8)
                vols.append(crop.astype(np.float32))
                labels.append(sex_lookup[int(match.group(1))])
    
    return np.array(vols), np.array(labels).astype(np.float32)


def generate_distance_data(n_samples, vol_size=32, swapped=False, seed=None):
    """
    Generate synthetic data for distance classification task.
    Two dots in 3D space - classify based on their distance.
    
    Args:
        n_samples: Number of samples to generate
        vol_size: Size of 3D volume
        swapped: If True, swap left/right position (for distribution shift)
        seed: Random seed
        
    Returns:
        X: Tensor of shape (n_samples, 1, vol_size, vol_size, vol_size)
        Y: Tensor of labels (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    X = torch.zeros(n_samples, 1, vol_size, vol_size, vol_size)
    Y = torch.zeros(n_samples, dtype=torch.long)
    
    for i in range(n_samples):
        # Randomly choose class
        label = np.random.randint(0, 2)
        
        # Set distance based on class
        if label == 0:  # Close
            distance = np.random.uniform(3, 8)
        else:  # Far
            distance = np.random.uniform(12, 18)
        
        # Random center point
        center = np.random.uniform(vol_size * 0.3, vol_size * 0.7, 3)
        
        # Random direction
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        # Calculate two points
        point1 = center - direction * distance / 2
        point2 = center + direction * distance / 2
        
        # Swap positions if requested (for distribution shift)
        if swapped:
            if point1[0] < vol_size / 2:  # If left
                point1[0] = vol_size - point1[0]  # Move to right
            if point2[0] < vol_size / 2:
                point2[0] = vol_size - point2[0]
        
        # Clip to volume bounds
        point1 = np.clip(point1, 0, vol_size - 1).astype(int)
        point2 = np.clip(point2, 0, vol_size - 1).astype(int)
        
        # Place dots
        X[i, 0, point1[0], point1[1], point1[2]] = 1
        X[i, 0, point2[0], point2[1], point2[2]] = 1
        Y[i] = label
    
    return X, Y


def generate_pattern_data(n_samples, img_size=64, pattern_type='checkerboard', 
                          noise_level=0, seed=None):
    """
    Generate synthetic 2D pattern data.
    
    Args:
        n_samples: Number of samples
        img_size: Image size (square)
        pattern_type: Type of pattern ('checkerboard', 'stripes', 'dots')
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed
        
    Returns:
        X: Array of images (n_samples, img_size, img_size)
        Y: Array of labels (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.zeros((n_samples, img_size, img_size))
    Y = np.zeros(n_samples, dtype=np.int64)
    
    for i in range(n_samples):
        label = np.random.randint(0, 2)
        
        if pattern_type == 'checkerboard':
            # Create checkerboard pattern
            freq = 4 if label == 0 else 8
            x, y = np.meshgrid(np.arange(img_size), np.arange(img_size))
            pattern = ((x // (img_size // freq)) + (y // (img_size // freq))) % 2
            
        elif pattern_type == 'stripes':
            # Create stripe pattern
            freq = 4 if label == 0 else 8
            x = np.arange(img_size)
            pattern = np.tile((x // (img_size // freq)) % 2, (img_size, 1))
            
        elif pattern_type == 'dots':
            # Create dot pattern
            spacing = 8 if label == 0 else 4
            pattern = np.zeros((img_size, img_size))
            for xi in range(0, img_size, spacing):
                for yi in range(0, img_size, spacing):
                    if xi < img_size and yi < img_size:
                        pattern[xi, yi] = 1
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        # Add noise
        if noise_level > 0:
            pattern = pattern + np.random.randn(img_size, img_size) * noise_level
        
        X[i] = pattern
        Y[i] = label
    
    return X, Y


def add_gaussian_noise_3d(volume, noise_level):
    """Add Gaussian noise to a 3D volume."""
    noise = np.random.randn(*volume.shape) * noise_level
    return volume + noise


def smooth_volume(volume, sigma=1.0):
    """Apply Gaussian smoothing to a 3D volume."""
    return gaussian_filter(volume, sigma=sigma)



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

                # Force Alignment
                img_nib = nib.load(os.path.join(nii_dir, f))
                mask_nib = nib.load(mask_path)

                # Force mask into QSM physical space
                if img_nib.shape == mask_nib.shape:
                    mask_nib = nib.Nifti1Image(mask_nib.get_fdata(), img_nib.affine, img_nib.header)

                img_obj = nib.as_closest_canonical(img_nib)
                mask_obj = nib.as_closest_canonical(mask_nib)
                
                raw_data = img_obj.get_fdata()
                mask_data = mask_obj.get_fdata()

                # Native Anchor Calculation (Disambiguate CHH vs MSW)
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

                # Strip out non-target labels
                mask_data = np.where(np.isin(mask_data, anchor_labels), mask_data, 0)
                anchor_mask = (mask_data > 0).astype(np.float32)

                if np.sum(anchor_mask) == 0:
                    print(f"Skipping {sub_id_int}: Target ROI not found after filtering.")
                    continue

                # Conditional Crop & Resample
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

                # Visualization
                if loaded_count == 0:
                    print(f"DEBUG: Showing orientation for Subject {sub_id_int}...")
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    mid_z = standardized_data.shape[2] // 2
                    ax[0].imshow(np.rot90(standardized_data[:, :, mid_z]), cmap='gray')
                    ax[0].set_title(f"QSM Crop (Sub {sub_id_int})")
                    ax[1].imshow(np.rot90(m_patch[:, :, mid_z]), cmap='jet')
                    ax[1].set_title("Target ROI (Filtered)")
                    plt.show()

                # Standardization
                standardized_data[standardized_data <= -2000] = 0.0
                if self.ext_mask_dir: 
                    skull_mask_path = os.path.join(self.ext_mask_dir, f"mag_{case_id}BrainExtractionMask.nii.gz")
                    if os.path.exists(skull_mask_path):
                        skull_mask = mask_crop_fn(nib.load(skull_mask_path).get_fdata(), anchor_mask, target_shape)
                        standardized_data = standardized_data * (skull_mask > 0)

                standardized_data = np.clip(standardized_data, -250.0, 250.0) / 250.0

                # Axial alignment
                processed_vol = standardized_data.astype(np.float32) 
                
                # Store
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
            img_slice = vol[:, :, meta['slice_idx']]
            img_tensor = torch.from_numpy(img_slice).unsqueeze(0) 
            
            clin_data = self.clinical_dict.get(str(meta['sub_id']), np.zeros(9))
            return img_tensor, torch.tensor(clin_data, dtype=torch.float32), int(meta['label']), index
    

def fast_resample_sharp(volume, msw_res=0.5, chh_res=0.9, target_shape=(64, 64, 72), sharpen=True):
    m_res, c_res = float(msw_res), float(chh_res)
    t_shape = target_shape if target_shape else (64, 64, 72)
    
    scaling_factor = m_res / c_res  
    in_center = np.array(volume.shape) / 2.0
    out_center = np.array(t_shape) / 2.0
    offset = in_center - (scaling_factor * out_center)
    
    # Resample with 'nearest' to fill the frame
    resampled = affine_transform(
        volume,
        matrix=scaling_factor * np.eye(3),
        offset=offset,
        output_shape=t_shape,
        order=3,
        mode='nearest' 
    )
    
    if sharpen:
        blurred = gaussian_filter(resampled, sigma=0.8)
        resampled = resampled + 0.5 * (resampled - blurred)

    # This creates a mask that is 1.0 in the center and fades to 0.0 at edges
    # avoiding the "hard zero" that causes the zero-padding artifact.
    mask = np.ones(t_shape)
    # Use a large-sigma Gaussian to create a smooth falloff at the very edges
    mask = gaussian_filter(mask, sigma=1.5) 
    # Normalize mask so center is 1.0
    mask = mask / mask.max()

    return (resampled * mask).astype(np.float32)


class BalancedDataset(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
        self.pos_idx = np.where(Y == 1)[0]
        self.neg_idx = np.where(Y == 0)[0]
        self.count = max(len(self.pos_idx), len(self.neg_idx)) * 2

    def __len__(self): return self.count

    def __getitem__(self, idx):
        i = np.random.choice(self.pos_idx) if idx % 2 == 0 else np.random.choice(self.neg_idx)
        return self.X[i], self.Y[i]