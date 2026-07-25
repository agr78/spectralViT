import os
import tempfile
import shutil
import gzip
import numpy as np
import pandas as pd
import nibabel as nib
from loader import QSM_Dataset
from loader import fast_resample_sharp

def prepare_qsm_dataset(source_type, nii_path, seg_path, csv_path, cache_path, 
                        load_cache=False, limit=None, mask_crop_fn=None,
                        unlabeled_nii_path=None, ext_mask_dir=None, cv_pad=False,
                        debug=False):

    print(f"\n{'='*60}")
    print(f"Preparing {source_type} dataset (Forced 6-dim alignment) {'[DEBUG MODE]' if debug else ''}")
    print(f"{'='*60}")
    
    # 1. PRE-FLIGHT LIVE STATS
    print(f"Pre-flight check: Validating {source_type} CSV mapping...")
    try:
        if source_type.upper() == 'CHH':
            df_v = pd.read_csv(csv_path, header=None).iloc[2:]
            v_matrix = df_v[[1, 2, 3, 4, 9, 10]].apply(pd.to_numeric, errors='coerce').dropna().values
        else:
            df_v = pd.read_csv(csv_path, header=1)
            df_v.columns = [str(c).strip().replace('\n', ' ') for c in df_v.columns]
            df_v['Sex_Num'] = df_v['Sex'].apply(lambda x: 1 if 'f' in str(x).lower() or '1' in str(x) else 0)
            v_matrix = df_v[['Age', 'Sex_Num', 'Disease Duration (year)', 
                             'pre op levadopa equivalent dose (mg)', 
                             'OFF (pre-dbs updrs)', 'ON (pre-dbs updrs)']].apply(pd.to_numeric, errors='coerce').fillna(0).values
        
        means = np.mean(v_matrix, axis=0)
        labels = ['Age', 'Sex', 'Dur', 'LEDD', 'Off-Pre', 'On-Pre']
        print(f"--- {source_type} CSV RAW MEANS ---")
        for i in range(6):
            print(f"  > {labels[i]:<8}: {means[i]:.2f}")
    except Exception as e:
        print(f"  ⚠️ Pre-flight mean calculation failed: {e}")

    # 2. DIRECTORY SETUP & CHH INJECTION LOGIC
    forced_unlabeled_ids = set()
    resample_fn = None
    
    if source_type.upper() == 'CHH':
        resample_fn = fast_resample_sharp 
        final_nii_dir, final_seg_dir = tempfile.mkdtemp(), tempfile.mkdtemp()
        
        # --- Process labeled path (ROI-Optional) ---
        all_files = [f for f in os.listdir(nii_path) if f.endswith('.nii.gz') and f[0].isdigit()]
        for f in all_files:
            try:
                sub_id_int = int(f.split('_')[0])
                if not debug:
                    os.symlink(os.path.join(nii_path, f), os.path.join(final_nii_dir, f"qsm_{sub_id_int}.nii.gz"))
                    roi_src = os.path.join(seg_path, f"{sub_id_int:02d}_roi_combined.nii")
                    if os.path.exists(roi_src):
                        with open(roi_src, 'rb') as f_in, gzip.open(os.path.join(final_seg_dir, f"seg_{sub_id_int:02d}.nii.gz"), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
            except Exception: continue

        # --- GENERATE CONSENSUS BBOX FROM LABELED DATA ---
        mean_min, mean_max = None, None
        existing_masks = [os.path.join(final_seg_dir, f) for f in os.listdir(final_seg_dir) if f.endswith('.nii.gz')]
        
        if existing_masks:
            try:
                bboxes = []
                for m in existing_masks:
                    m_data = nib.load(m).get_fdata()
                    coords = np.argwhere(m_data > 0)
                    if coords.size > 0:
                        bboxes.append((coords.min(axis=0), coords.max(axis=0)))
                
                if bboxes:
                    mean_min = np.mean([b[0] for b in bboxes], axis=0).astype(int)
                    mean_max = np.mean([b[1] for b in bboxes], axis=0).astype(int)
                    print(f"Calculated consensus BBox: Min{mean_min} Max{mean_max}")
            except Exception as e:
                print(f"Warning: BBox calculation failed: {e}")

        # --- Process Unlabeled Path with BBox Mask Injection ---
        print(f"Checking unlabeled path: {unlabeled_nii_path} (Exists: {os.path.exists(unlabeled_nii_path) if unlabeled_nii_path else False})")
        if unlabeled_nii_path and os.path.exists(unlabeled_nii_path):
            extra_files = [f for f in os.listdir(unlabeled_nii_path) if f.endswith('.nii.gz')]
            for f in extra_files:
                try:
                    sub_id_str = f.split('_')[0].lstrip('0') 
                    sub_id_int = int(sub_id_str) if sub_id_str != "" else 0 
                    forced_unlabeled_ids.add(str(sub_id_int))
                    if not debug:
                        target_link = os.path.join(final_nii_dir, f"qsm_{sub_id_int}.nii.gz")
                        if not os.path.exists(target_link):
                            img_src = os.path.join(unlabeled_nii_path, f)
                            os.symlink(img_src, target_link)
                            
                            # Inject consensus mask for the unlabeled subject
                            if mean_min is not None:
                                unl_img = nib.load(img_src)
                                box_mask = np.zeros(unl_img.shape, dtype=np.float32)
                                # Clamp indices to actual image dimensions
                                z_max = min(mean_max[2], unl_img.shape[2])
                                box_mask[mean_min[0]:mean_max[0], mean_min[1]:mean_max[1], mean_min[2]:z_max] = 1.0
                                
                                target_mask = os.path.join(final_seg_dir, f"seg_{sub_id_int:02d}.nii.gz")
                                nib.save(nib.Nifti1Image(box_mask, unl_img.affine, unl_img.header), target_mask)
                except Exception: continue
    else:
        final_nii_dir, final_seg_dir = nii_path, seg_path

    # --- DISCOVER FILES ON DISK ---
    if debug and source_type.upper() == 'CHH':
        all_subject_ids_on_disk = set()
        for f in os.listdir(nii_path):
             if f.endswith('.nii.gz') and f[0].isdigit():
                 all_subject_ids_on_disk.add(str(int(f.split('_')[0])))
        for sid in forced_unlabeled_ids:
            all_subject_ids_on_disk.add(sid)
    else:
        all_qsm_files = [f for f in os.listdir(final_nii_dir) if f.endswith('.nii.gz')]
        all_subject_ids_on_disk = []
        for f in all_qsm_files:
            try:
                parts = f.replace('qsm_', '').split('_')[0].split('.')[0]
                all_subject_ids_on_disk.append(str(int(float(parts))))
            except: continue
        all_subject_ids_on_disk = set(all_subject_ids_on_disk)

    # 4. CLINICAL DICTIONARY
    clinical_dict, label_map = {}, {}
    clin_dim = 6 
    
    if source_type.upper() == 'CHH':
        df_full = pd.read_csv(csv_path, header=None)
        for _, row in df_full.iloc[2:].iterrows():
            try:
                sub_id = str(int(float(row[0])))
                vec = [row[1], row[2], row[3], row[4], row[9], row[10]]
                clinical_dict[sub_id] = np.nan_to_num(np.array(vec, dtype=np.float32))
                if sub_id in forced_unlabeled_ids:
                    label_map[sub_id] = -1
                else:
                    off_p = pd.to_numeric(row[9], errors='coerce')
                    post_p = pd.to_numeric(row[11], errors='coerce') 
                    label_map[sub_id] = 1 if (not pd.isna(off_p) and not pd.isna(post_p) and off_p > 0 and (off_p - post_p)/off_p >= 0.30) else (0 if not pd.isna(off_p) and not pd.isna(post_p) else -1)
            except: continue
    else:
        motor_df = pd.read_csv(csv_path, header=1)
        motor_df.columns = [str(c).strip().replace('\n', ' ') for c in motor_df.columns]
        for _, row in motor_df.iterrows():
            try:
                raw_id = row.get('CORNELL ID')
                if pd.isna(raw_id): continue
                sub_id = str(int(float(raw_id)))
                vec = [row.get('Age'), 1 if 'f' in str(row.get('Sex')).lower() or '1' in str(row.get('Sex')) else 0,
                       row.get('Disease Duration (year)'), row.get('pre op levadopa equivalent dose (mg)'),
                       row.get('OFF (pre-dbs updrs)'), row.get('ON (pre-dbs updrs)')]
                clinical_dict[sub_id] = np.nan_to_num(np.array(vec, dtype=np.float32))
                off_p = pd.to_numeric(row.get('OFF (pre-dbs updrs)'), errors='coerce')
                post_p = pd.to_numeric(row.get('OFF meds ON stim 6mo'), errors='coerce')
                label_map[sub_id] = 1 if (not pd.isna(off_p) and not pd.isna(post_p) and off_p > 0 and (off_p - post_p)/off_p >= 0.30) else (0 if not pd.isna(off_p) and not pd.isna(post_p) else -1)
            except: continue

    # --- UNLABELED CATCH-ALL ---
    missing_from_csv = []
    for sid in all_subject_ids_on_disk:
        if sid not in clinical_dict:
            clinical_dict[sid] = np.zeros(clin_dim, dtype=np.float32)
            missing_from_csv.append(sid)
        if sid not in label_map:
            label_map[sid] = -1 

    # 6. BREAKDOWN CALCULATION
    all_final_ids = list(all_subject_ids_on_disk)
    lbls = [label_map[sid] for sid in all_final_ids]
    valid_vecs = np.array([clinical_dict[sid] for sid in all_final_ids if not np.all(clinical_dict[sid] == 0)])
    
    print(f"\nFinal {source_type} Breakdown:")
    print(f" - Unique Subjects on Disk: {len(all_final_ids)}")
    print(f" - Labeled Responders (1): {lbls.count(1)}")
    print(f" - Labeled Non-Responders (0): {lbls.count(0)}")
    print(f" - Unlabeled subjects (-1): {lbls.count(-1)}")
    
    if valid_vecs.size > 0:
        f_means = np.mean(valid_vecs, axis=0)
        labels = ['Age', 'Sex', 'Dur', 'LEDD', 'Off-Pre', 'On-Pre']
        print(f" - Verified Realized Means (Matched Data Only):")
        for i in range(6):
            print(f"    > {labels[i]:<8}: {f_means[i]:.2f}")
    
    if missing_from_csv:
        print(f"\n❌ FULL MISSING LIST ({len(missing_from_csv)} subjects):")
        try:
            sorted_missing = sorted([int(x) for x in missing_from_csv])
            print(f"  {sorted_missing}")
        except:
            print(f"  {sorted(missing_from_csv)}")
            
    if debug:
        print(f"\n[DEBUG] Terminating early as requested. No caching performed.")
        return None

    # 5. DATASET INIT (Caching starts here)
    dataset = QSM_Dataset(final_nii_dir, final_seg_dir, mask_crop_fn, clinical_dict, label_map, 
                          limit=limit, cache_path=cache_path, load_cache=load_cache, 
                          return_index=True, ext_mask_dir=ext_mask_dir, resample_fn=resample_fn)
    dataset.clin_dim = clin_dim
    return dataset

def compare_datasets(ds1, ds2):
    if ds1 is None or ds2 is None:
        print("\nCannot compare: One or both datasets are in Debug mode (None).")
        return
    labels = ['Age', 'Sex', 'Dur', 'LEDD', 'Off-Pre', 'On-Pre']
    v1 = np.array([ds1.clinical_dict[sid] for sid in ds1.clinical_dict.keys() if not np.all(ds1.clinical_dict[sid] == 0)])
    v2 = np.array([ds2.clinical_dict[sid] for sid in ds2.clinical_dict.keys() if not np.all(ds2.clinical_dict[sid] == 0)]) 
    print(f"\n{'Variable':<10} | {'MSW (Matched)':>12} | {'CHH (Matched)':>12} | {'Status'}")
    print("-" * 65)
    for i in range(6):
        m1, m2 = np.mean(v1[:, i]), np.mean(v2[:, i])
        status = "✅ OK" if abs(m1 - m2) < 15 else "❌ MISALIGNED"
        if i == 1: status = "✅ OK" if abs(m1 - m2) < 0.3 else "❌ MISALIGNED"
        print(f"{labels[i]:<10} | {m1:>12.2f} | {m2:>12.2f} | {status}")