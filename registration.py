import os
import re
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm
import ants
from skimage.transform import resize
import pandas as pd

def get_clinical_metadata(CHH_CSV=None,MSW_CSV=None):
    clinical_dict, label_map = {}, {}
    # Load CHH
    if CHH_CSV is not None:
        try:
            df_chh = pd.read_csv(CHH_CSV, header=None)
            for _, row in df_chh.iloc[2:].iterrows():
                if pd.isna(row[0]): continue 
                sub_id = str(int(float(row[0])))
                vec = [row[1], row[2], row[3], row[4], row[9], row[10]]
                clinical_dict[sub_id] = np.nan_to_num(np.array(vec, dtype=np.float32))
                off_p, post_p = pd.to_numeric(row[9], errors='coerce'), pd.to_numeric(row[11], errors='coerce') 
                label_map[sub_id] = 1 if (not pd.isna(off_p) and not pd.isna(post_p) and off_p > 0 and (off_p - post_p)/off_p >= 0.30) else (0 if not pd.isna(off_p) and not pd.isna(post_p) else -1)
            print('Loaded CHH clinical dictionary')
        except Exception as e: print(f"!! CHH CSV Error: {e}")
    # Load MSW
    if MSW_CSV is not None:
        try:
            df_msw = pd.read_csv(MSW_CSV, header=1)
            df_msw.columns = [str(c).strip().replace('\n', ' ') for c in df_msw.columns]
            for _, row in df_msw.iterrows():
                raw_id = row.get('CORNELL ID')
                if pd.isna(raw_id): continue
                sub_id = str(int(float(raw_id)))
                sex_num = 1 if 'f' in str(row.get('Sex')).lower() or '1' in str(row.get('Sex')) else 0
                vec = [row.get('Age'), sex_num, row.get('Disease Duration (year)'), row.get('pre op levadopa equivalent dose (mg)'), row.get('OFF (pre-dbs updrs)'), row.get('ON (pre-dbs updrs)')]
                clinical_dict[sub_id] = np.nan_to_num(np.array(vec, dtype=np.float32))
                off_p, post_p = pd.to_numeric(row.get('OFF (pre-dbs updrs)'), errors='coerce'), pd.to_numeric(row.get('OFF meds ON stim 6mo'), errors='coerce')
                label_map[sub_id] = 1 if (not pd.isna(off_p) and not pd.isna(post_p) and off_p > 0 and (off_p - post_p)/off_p >= 0.30) else (0 if not pd.isna(off_p) and not pd.isna(post_p) else -1)
            print('Loaded MSW clinical dictionary')
        except Exception as e: print(f"!! MSW Error: {e}")
    return clinical_dict, label_map

def get_processed_data(f, img_dir, seg_dir, target_shape=None):
    match = re.search(r'\d+', f)
    if not match: return None, None, None
    sub_id_int = int(match.group())
    
    mask_name = f"seg_{sub_id_int:02d}.nii.gz"
    mask_path = os.path.join(seg_dir, mask_name)
    if not os.path.exists(mask_path) and os.path.exists(mask_path + ".gz"): mask_path += ".gz"
    if not os.path.exists(mask_path): return None, None, sub_id_int

    # Load and clean
    img_nib = nib.load(os.path.join(img_dir, f))
    msk_nib = nib.load(mask_path)
    img_data = np.nan_to_num(img_nib.get_fdata())
    msk_data = np.nan_to_num(msk_nib.get_fdata())

    # Anchoring logic
    anchors = [5, 6]
    coords = np.argwhere(np.isin(msk_data, anchors))
    if len(coords) == 0: coords = np.argwhere(msk_data > 0)
    cz, cy, cx = coords.mean(axis=0).astype(int)

    # Crop
    hz, hy, hx = (40, 40, 40) # Smaller initial window for better centering
    img_f = img_data[max(0, cz-hz):cz+hz, max(0, cy-hy):cy+hy, max(0, cx-hx):cx+hx]
    msk_f = msk_data[max(0, cz-hz):cz+hz, max(0, cy-hy):cy+hy, max(0, cx-hx):cx+hx]

    # Zoom
    res_ratio = 0.5 / 0.9
    img_f = zoom(img_f, (res_ratio, res_ratio, 1.0), order=1)
    msk_f = zoom(msk_f, (res_ratio, res_ratio, 1.0), order=0)

    # Resize to Target
    if target_shape:
        img_f = resize(img_f, target_shape, order=1, preserve_range=True)
        msk_f = resize(msk_f, target_shape, order=0, preserve_range=True)

    # Intensity Normalization (Crucial for ANTs stability)
    img_f = (img_f - np.min(img_f)) / (np.max(img_f) - np.min(img_f) + 1e-7)

    return img_f, msk_f, sub_id_int

# --- Build Template ---
def build_stable_template(files, img_dir, seg_dir, target_shape_out):
    vols = []
    print(f">>> Standardizing {len(files)} volumes...")
    for f in tqdm(files):
        vol_data, _, _ = get_processed_data(f, img_dir, seg_dir, target_shape=target_shape_out)
        if vol_data is not None:
            # Force identity orientation/origin to prevent coordinate explosions
            ants_img = ants.from_numpy(vol_data.astype(np.float32))
            ants_img.set_spacing((1.0, 1.0, 1.0))
            ants_img.set_origin((0, 0, 0))
            vols.append(ants_img)
    
    print(">>> Running Template Builder (Translation-first for stability)...")
    # We use a very low number of iterations and "Translation" to ensure it doesn't crash
    tmpl = ants.build_template(initial_template=None, 
                               image_list=vols, 
                               iterations=1, 
                               type_of_transform='Translation')
    return tmpl

# --- Parallel Worker for Loading ---
def get_processed_data_worker(f, img_dir, seg_dir, clinical_dict, label_map, EXCLUDE_IDS, target_shape=None):
    print(f">>> Loading and clinical mapping (excluding {EXCLUDE_IDS})...")
    match = re.search(r'\d+', f)
    if not match: return None
    sub_id_int = int(match.group())
    sub_id_str = str(sub_id_int)
    if sub_id_int in EXCLUDE_IDS: return None
    
    mask_name = f"seg_{sub_id_int:02d}.nii.gz"
    mask_path = os.path.join(seg_dir, mask_name)
    if not os.path.exists(mask_path) and os.path.exists(mask_path + ".gz"): mask_path += ".gz"
    if not os.path.exists(mask_path): return None

    try:
        img_data = np.nan_to_num(nib.load(os.path.join(img_dir, f)).get_fdata())
        msk_data = np.nan_to_num(nib.load(mask_path).get_fdata())

        anchors = [5, 6]
        coords = np.argwhere(np.isin(msk_data, anchors))
        if len(coords) == 0: coords = np.argwhere(msk_data > 0)
        cz, cy, cx = coords.mean(axis=0).astype(int)

        hz, hy, hx = (40, 40, 40) 
        img_f = img_data[max(0, cz-hz):cz+hz, max(0, cy-hy):cy+hy, max(0, cx-hx):cx+hx]
        
        # --- FIXED: Use B-Spline (order=3) for smooth yet sharp intensity transitions ---
        res_ratio = 0.5 / 0.9
        img_f = zoom(img_f, (res_ratio, res_ratio, 1.0), order=3)
        if target_shape:
            img_f = resize(img_f, target_shape, order=3, preserve_range=True)

        raw_ppb = img_f.copy()
        p2, p98 = np.percentile(img_f, [2, 98])
        norm_img = np.clip((img_f - p2) / (p98 - p2 + 1e-7), 0, 1)

        return {
            'norm': norm_img, 
            'raw': raw_ppb, 
            'id': sub_id_int,
            'lab': label_map.get(sub_id_str, -1),
            'vec': clinical_dict.get(sub_id_str, None)
        }
    except Exception:
        return None

# --- Parallel Worker for Registration ---
def register_worker(data, target_template_numpy):
    target_template = ants.from_numpy(target_template_numpy.astype(np.float32))
    target_template.set_spacing((1.0, 1.0, 1.0))
    target_template.set_origin((0, 0, 0))

    moving_norm = ants.from_numpy(data['norm'].astype(np.float32))
    moving_raw = ants.from_numpy(data['raw'].astype(np.float32))
    
    for img in [moving_norm, moving_raw]:
        img.set_spacing((1.0, 1.0, 1.0))
        img.set_origin((0, 0, 0))
    
    reg = ants.registration(fixed=target_template, moving=moving_norm, type_of_transform='Rigid')
    
    # --- FIXED: Use bSpline for the final warp to ensure high-quality alignment ---
    warped_raw = ants.apply_transforms(
        fixed=target_template, 
        moving=moving_raw, 
        transformlist=reg['fwdtransforms'],
        interpolator='bSpline'
    )
    
    return {
        'id': data['id'],
        'lab': data['lab'],
        'vec': data['vec'],
        'native_raw': data['raw'],
        'warped_raw': warped_raw.numpy()
    }

def get_processed_data_worker_chh(f, img_dir, seg_dir, clinical_dict, label_map, EXCLUDE_IDS, target_shape=None):
    print(f">>> Loading and clinical mapping (excluding {EXCLUDE_IDS})...")
    match = re.search(r'\d+', f)
    if not match: return None
    sub_id_int = int(match.group())
    sub_id_str = str(sub_id_int)
    
    if sub_id_int in EXCLUDE_IDS: return None
    
    mask_name = f"{sub_id_int:02d}_roi_combined.nii"
    if not os.path.exists(os.path.join(seg_dir, mask_name)):
        mask_name = f"{sub_id_int}_roi_combined.nii"
    
    mask_path = os.path.join(seg_dir, mask_name)
    if not os.path.exists(mask_path) and os.path.exists(mask_path + ".gz"): mask_path += ".gz"
    if not os.path.exists(mask_path): return None

    try:
        img_nib = nib.load(os.path.join(img_dir, f))
        img_data = np.nan_to_num(img_nib.get_fdata())
        
        # Zero out background padding
        img_data[img_data < -1000] = 0 
        
        msk_data = np.nan_to_num(nib.load(mask_path).get_fdata())

        anchors = [1, 4]
        coords = np.argwhere(np.isin(msk_data, anchors))
        if len(coords) == 0: coords = np.argwhere(msk_data > 0)
        cz, cy, cx = coords.mean(axis=0).astype(int)

        # --- UPDATED: Standardized FOV (Zoomed in for CHH) ---
        # MSW (hz=40 @ 0.5mm) = 20mm radius. 
        # CHH (hz=22 @ 0.9mm) ≈ 20mm radius.
        hz, hy, hx = (22, 22, 22) 
        img_f = img_data[max(0, cz-hz):cz+hz, max(0, cy-hy):cy+hy, max(0, cx-hx):cx+hx]
        
        if target_shape:
            img_f = resize(img_f, target_shape, order=3, preserve_range=True)

        raw_ppb = img_f.copy()
        p2, p98 = np.percentile(img_f, [2, 98])
        norm_img = np.clip((img_f - p2) / (p98 - p2 + 1e-7), 0, 1)

        return {
            'norm': norm_img, 
            'raw': raw_ppb, 
            'id': sub_id_int,
            'lab': label_map.get(sub_id_str, -1),
            'vec': clinical_dict.get(sub_id_str, None)
        }
    except Exception:
        return None