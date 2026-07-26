import torch
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import affine_transform, gaussian_filter
from util import robust_flatten
import pandas as pd
import re
import glob
import io

def get_slice_level_data(dataset, target_dim, include_unlabeled=False):
    all_imgs, all_clins, all_lbls, subj_map = [], [], [], []
    for i in range(len(dataset)):
        img, clin, lbl, _ = dataset[i]
        if (i % 72) in [0]: continue  # Skip specific slices if necessary
        if (lbl != -1) or (include_unlabeled and lbl == -1):
            all_imgs.append(robust_flatten(img.numpy().squeeze(), target_dim))
            all_clins.append(clin.numpy())
            all_lbls.append(lbl)
            subj_map.append(i)
    return np.array(all_imgs), np.array(all_clins), np.array(all_lbls), np.array(subj_map)

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


def _parse_age_value(val, age_in_months=False):
    """
    Parse a single age entry that may be:
      - a plain number (e.g. 34, 34.0, 408 if in months)
      - a 5-year bracket string as used by HCP-YA's unrestricted release ('22-25')
      - an open-ended bracket ('36+')
    Returns a float age in years, or np.nan if it can't be parsed.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        v = float(val)
    else:
        s = str(val).strip()
        m = re.match(r'^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$', s)
        if m:
            v = (float(m.group(1)) + float(m.group(2))) / 2.0
        else:
            m = re.match(r'^(\d+(?:\.\d+)?)\s*\+$', s)
            if m:
                # Open-ended bracket (e.g. "36+"): assume the same ~5-yr width as
                # the rest of the HCP-YA scale so it stays roughly comparable.
                v = float(m.group(1)) + 2.5
            else:
                m = re.match(r'^(\d+(?:\.\d+)?)', s)
                v = float(m.group(1)) if m else np.nan
    if np.isnan(v):
        return v
    return v / 12.0 if age_in_months else v


def load_hcp_data(data_dir, csv_path, vol_size=96,
                   file_glob='*T1w*restore*brain*.nii.gz',
                   id_col=None, age_col=None, age_in_months=False,
                   verbose=True):
    """
    Load HCP (Human Connectome Project) structural T1w volumes for AGE PREDICTION
    (continuous regression), the HCP counterpart of load_ixi_data's sex classification.

    Unlike IXI's flat directory of files, HCP releases nest each subject's scan a
    few levels deep and use different naming conventions per release, e.g.:
      - HCP-YA (MNINonLinear):  <data_dir>/<subject>/MNINonLinear/T1w_restore_brain.nii.gz
      - HCP-YA (native/ACPC):   <data_dir>/<subject>/T1w/T1w_acpc_dc_restore_brain.nii.gz
      - HCP-Aging/Development (BIDS-derivatives): <data_dir>/sub-*/anat/*_desc-preproc_T1w.nii.gz
    so this loader searches `data_dir` recursively for `file_glob` and matches each
    file back to a subject by looking for the CSV's subject ID as a path token,
    rather than assuming a fixed folder depth like the IXI loader does.

    Age handling also differs from IXI's binary sex label: HCP-YA's unrestricted
    CSV only ships 5-year brackets ("22-25", "26-30", "31-35", "36+"), while
    HCP-Aging/Development and the restricted HCP-YA CSV give an exact age (in
    years, or months under a NDA-style "interview_age" column). This function
    auto-detects which of those it's given and converts everything to a single
    continuous "age in years" float so the rest of the pipeline (PCA, ViT/U-Net
    regression heads, MSE/MAE losses) doesn't need to know the difference.
    Brackets are a real limitation, though: they cap the achievable MAE at roughly
    half the bracket width, so prefer an exact-age column/CSV when you have one.

    Args:
        data_dir: Root directory to search recursively for T1w NIfTI volumes.
        csv_path: Path to CSV with a subject-ID column and an age column.
        vol_size: Size of the cropped cubic volume (must match the model's
            expected input, e.g. 96 to match SpectralViT/SpatialViT defaults).
        file_glob: Recursive glob (relative to data_dir) used to find volumes.
            Adjust to match your preprocessing pipeline, e.g.:
              'T1w_restore_brain.nii.gz'            (HCP-YA, MNINonLinear space)
              'T1w_acpc_dc_restore_brain.nii.gz'    (HCP-YA, native/ACPC space)
              '*_desc-preproc_T1w.nii.gz'           (HCP-Aging/Development, BIDS)
        id_col: Subject-ID column name in the CSV. Auto-detected if None.
        age_col: Age column name in the CSV. Auto-detected if None.
        age_in_months: Set True if age_col is in months (e.g. NDA "interview_age").
            Auto-set True if a column literally named "interview_age" is selected.
        verbose: Print auto-detected columns and loading progress.

    Returns:
        X: Array of volumes (N, vol_size, vol_size, vol_size), per-volume z-scored.
        Y: Array of ages in years (N,), dtype float32.
        subject_ids: List of the N subject ID strings, in the same order as X/Y.
    """
    df = pd.read_csv(csv_path)

    if id_col is None:
        # NDA exports carry two subject identifiers: 'subjectkey' (the global,
        # cross-study de-identified GUID) and 'src_subject_id' (the site-specific
        # ID, e.g. "HCA6002236"). Imaging package directories are named after the
        # latter, so it must win if both are present.
        cols_upper = {c.upper(): c for c in df.columns}
        if 'SRC_SUBJECT_ID' in cols_upper:
            id_col = cols_upper['SRC_SUBJECT_ID']
        else:
            id_candidates = [c for c in df.columns if 'SUBJ' in c.upper() or c.upper() == 'ID']
            if not id_candidates:
                id_candidates = [c for c in df.columns if 'ID' in c.upper()]
            if not id_candidates:
                raise ValueError("Could not auto-detect a subject-ID column; pass id_col explicitly.")
            id_col = id_candidates[0]

    if age_col is None:
        age_candidates = [c for c in df.columns if 'AGE' in c.upper()]
        if not age_candidates:
            raise ValueError("Could not auto-detect an age column; pass age_col explicitly.")
        yrs = [c for c in age_candidates if 'YR' in c.upper() or 'YEAR' in c.upper()]
        interview = [c for c in age_candidates if 'INTERVIEW' in c.upper()]
        if yrs:
            age_col = yrs[0]
        elif interview:
            age_col = interview[0]
            age_in_months = True
        else:
            age_col = age_candidates[0]

    if verbose:
        unit = " (months)" if age_in_months else ""
        print(f"Using ID column: '{id_col}', age column: '{age_col}'{unit}")

    age_lookup = {
        str(sid).strip(): _parse_age_value(age, age_in_months)
        for sid, age in zip(df[id_col], df[age_col])
    }
    age_lookup = {k: v for k, v in age_lookup.items() if not np.isnan(v)}
    id_lookup_upper = {k.upper(): k for k in age_lookup}

    files = sorted(glob.glob(os.path.join(data_dir, '**', file_glob), recursive=True))
    if verbose:
        print(f"Found {len(files)} candidate volumes under {data_dir}")

    vols, labels, subj_ids = [], [], []
    for fpath in tqdm(files, desc="Loading HCP Data"):
        rel = os.path.relpath(fpath, data_dir)
        tokens = re.split(r'[\\/_.\-]', rel)

        matched_id = None
        for tok in tokens:
            if tok.upper() in id_lookup_upper:
                matched_id = id_lookup_upper[tok.upper()]
                break
        if matched_id is None:
            # Fallback for flat/numeric layouts (e.g. IXI-style): a bare run of digits.
            digit_match = re.search(r'(\d{5,})', rel)
            if digit_match and digit_match.group(1) in age_lookup:
                matched_id = digit_match.group(1)
        if matched_id is None:
            continue

        try:
            img = nib.load(fpath).get_fdata()
        except Exception as e:
            if verbose:
                print(f"Skipping {fpath}: {e}")
            continue

        c = np.array(img.shape[:3]) // 2
        r = vol_size // 2
        crop = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r, c[2]-r:c[2]+r]
        if crop.shape != (vol_size, vol_size, vol_size):
            continue

        crop = (crop - np.mean(crop)) / (np.std(crop) + 1e-8)
        vols.append(crop.astype(np.float32))
        labels.append(age_lookup[matched_id])
        subj_ids.append(matched_id)

    if verbose:
        print(f"Loaded {len(vols)} / {len(files)} volumes with matched ages")

    return np.array(vols), np.array(labels).astype(np.float32), subj_ids


def prepare_nda_csv(txt_path, out_csv_path=None, sep='\t'):
    """
    Convert an NDA "data structure" export (the .txt file bundled in a downloadcmd
    package, e.g. demographics01.txt) into a plain CSV that load_hcp_data can read.

    NDA's structured-data text files follow a documented but pandas-unfriendly
    layout that a plain pd.read_csv() will misparse:
      line 1: "<data_structure_short_name>,<version>"  -- not data, e.g. "ndar_subject01,1"
      line 2: short column names                        -- the header we actually want
      line 3: long-form column descriptions              -- not data
      line 4+: actual rows, tab-separated

    Rather than hardcoding those line numbers (some exports omit the marker line,
    or the description row), both are detected heuristically: the marker line is
    identified by having no field separator, and the description row by having a
    much higher average word-count per field than real data does. A plain
    CSV/TSV with no preamble at all passes through unchanged.

    Args:
        txt_path: Path to the raw NDA .txt export.
        out_csv_path: Where to write the cleaned CSV. Defaults to
            "<txt_path without extension>_clean.csv".
        sep: Field separator used in the export (NDA uses tabs).

    Returns:
        The path to the cleaned CSV (out_csv_path).
    """
    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        raw_lines = f.readlines()
    raw_lines = [ln.rstrip('\n').rstrip('\r') for ln in raw_lines if ln.strip() != '']

    idx = 0
    # Marker line has no field separator but does have a comma, e.g. "ndar_subject01,1"
    if sep not in raw_lines[0] and ',' in raw_lines[0]:
        idx = 1

    header_fields = raw_lines[idx].split(sep)
    data_start = idx + 1

    if data_start < len(raw_lines):
        next_fields = raw_lines[data_start].split(sep)
        if len(next_fields) == len(header_fields):
            nonempty = [f for f in next_fields if f.strip()]
            avg_words = np.mean([len(f.split()) for f in nonempty]) if nonempty else 0
            if avg_words > 1.5:  # reads like prose -> this is the description row
                data_start += 1

    csv_text = sep.join(header_fields) + '\n' + '\n'.join(raw_lines[data_start:])
    df = pd.read_csv(io.StringIO(csv_text), sep=sep)

    if out_csv_path is None:
        out_csv_path = os.path.splitext(txt_path)[0] + '_clean.csv'
    df.to_csv(out_csv_path, index=False)
    print(f"Wrote cleaned CSV ({len(df)} rows, {len(df.columns)} cols) to {out_csv_path}")
    print(f"Columns: {list(df.columns)}")
    return out_csv_path


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