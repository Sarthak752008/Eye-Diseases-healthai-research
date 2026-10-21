"""
scripts/prepare_cataract_and_stye_datasets.py
---------------------------------------------
Audits and prepares leak-free manifests and patient/subject-isolated splits for:
1. Nuclear Cataract Dataset (Slit-lamp LOCS III graded images, Cruz-Vega et al. / Ignatowicz et al.)
2. Stye / Hordeolum Dataset (Eyelid margin lesion detection paired with normal anterior ocular controls)

Clinical Safety & Integrity:
- Zero data leakage: All images of the same patient/subject reside strictly in one partition (train, val, or test).
- Genuine ground-truth preservation.
- Full traceability with MD5 hashes, image dimensions, and original relative paths.
"""

import os
import sys
import zipfile
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

def compute_md5(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def prepare_cataract_manifest():
    print("\n" + "=" * 60)
    print("1. PREPARING NUCLEAR CATARACT DATASET MANIFEST")
    print("=" * 60)

    excel_path = os.path.join(WORKSPACE_ROOT, 'Nuclear Cataract Dataset', 'NC_images_used_for_evaluation.xlsx')
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Evaluation spreadsheet missing at: {excel_path}")

    # Read sheet via zero-dep parser
    with zipfile.ZipFile(excel_path, 'r') as z:
        shared_strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            tree = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in tree.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                t = si.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')
                shared_strings.append(t.text if t is not None else '')

        tree = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        rows = []
        for row in tree.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row'):
            row_vals = []
            for c in row.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c'):
                v = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                val = v.text if v is not None else ''
                t = c.get('t')
                if t == 's' and val.isdigit():
                    val = shared_strings[int(val)]
                row_vals.append(val)
            rows.append(row_vals)

    data_rows = rows[1:]
    print(f"Loaded {len(data_rows)} evaluation records from spreadsheet.")

    records = []
    missing_count = 0

    for r in data_rows:
        vol_str = r[0].strip()
        vol = str(int(vol_str)).zfill(2) if vol_str.isdigit() else vol_str
        case = r[1].strip()
        fname = r[2].strip()
        locs_grade = int(r[3].strip())

        # Resolve path
        candidates = [
            os.path.join(WORKSPACE_ROOT, 'Nuclear Cataract Dataset', vol, case, f"{fname}.JPG"),
            os.path.join(WORKSPACE_ROOT, 'Nuclear Cataract Dataset', vol, case, f"{fname}.jpg"),
            os.path.join(WORKSPACE_ROOT, 'Nuclear Cataract Dataset', vol, case.lower().capitalize(), f"{fname}.JPG"),
            os.path.join(WORKSPACE_ROOT, 'Nuclear Cataract Dataset', vol, case.lower().capitalize(), f"{fname}.jpg"),
            os.path.join(WORKSPACE_ROOT, 'Nuclear Cataract Dataset', vol, case.upper(), f"{fname}.JPG"),
            os.path.join(WORKSPACE_ROOT, 'Nuclear Cataract Dataset', vol, case.upper(), f"{fname}.jpg"),
        ]
        resolved = None
        for c in candidates:
            if os.path.exists(c):
                resolved = c
                break

        if resolved is None:
            missing_count += 1
            continue

        rel_path = os.path.relpath(resolved, WORKSPACE_ROOT).replace('\\', '/')
        md5_val = compute_md5(resolved)
        with Image.open(resolved) as im:
            w, h = im.size

        # Binary label: Grade 1 = Clear lens / Non-cataractous (0), Grades 2-6 = Nuclear Cataract (1)
        binary_label = 0 if locs_grade == 1 else 1
        class_name = "normal_clear_lens" if locs_grade == 1 else "nuclear_cataract"

        records.append({
            'image_id': f"cataract_{vol}_{case}_{fname}",
            'filename': os.path.basename(resolved),
            'relative_path': rel_path,
            'patient_id': f"patient_{vol}",
            'eye_laterality': 'OD' if case.upper() == 'DER' else ('OS' if case.upper() == 'IZQ' else case),
            'locs_grade': locs_grade,
            'label': binary_label,
            'class_name': class_name,
            'dataset_source': 'Nuclear_Cataract_Database_CruzVega_Valenciana',
            'modality': 'slit_lamp_biomicroscopy',
            'md5': md5_val,
            'width': w,
            'height': h,
            'aspect_ratio': round(w / h, 4)
        })

    df_cat = pd.DataFrame(records)
    print(f"Matched {len(df_cat)} records across {df_cat['patient_id'].nunique()} unique patients.")
    print("Grade breakdown:\n", df_cat['locs_grade'].value_counts().sort_index())
    print("Binary class breakdown:\n", df_cat['class_name'].value_counts())

    # Patient-isolated split (70% train, 15% val, 15% test)
    # Stratified at patient level by highest grade observed in that patient
    patient_max_grade = df_cat.groupby('patient_id')['locs_grade'].max().to_dict()
    unique_patients = sorted(list(patient_max_grade.keys()))

    np.random.seed(42)
    # Group patients by their max grade for stratified patient splitting
    grade_to_patients = {}
    for p, g in patient_max_grade.items():
        grade_to_patients.setdefault(g, []).append(p)

    train_patients, val_patients, test_patients = [], [], []
    for g, p_list in grade_to_patients.items():
        np.random.shuffle(p_list)
        n = len(p_list)
        if n == 1:
            train_patients.extend(p_list)
        elif n == 2:
            train_patients.append(p_list[0])
            test_patients.append(p_list[1])
        else:
            n_train = max(1, int(round(n * 0.70)))
            n_val = max(1, int(round(n * 0.15)))
            train_patients.extend(p_list[:n_train])
            val_patients.extend(p_list[n_train:n_train + n_val])
            test_patients.extend(p_list[n_train + n_val:])

    patient_to_split = {}
    for p in train_patients:
        patient_to_split[p] = 'train'
    for p in val_patients:
        patient_to_split[p] = 'val'
    for p in test_patients:
        patient_to_split[p] = 'test'

    df_cat['split'] = df_cat['patient_id'].map(patient_to_split)

    print("\nPatient-Isolated Split Results:")
    print("Patients per split:", Counter(patient_to_split.values()))
    print("Images per split:\n", df_cat['split'].value_counts())
    print("Split vs Binary Class:\n", pd.crosstab(df_cat['split'], df_cat['class_name']))

    manifest_out = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'nuclear_cataract_manifest.csv')
    os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
    df_cat.to_csv(manifest_out, index=False)
    print(f"Saved Cataract manifest to: {manifest_out}")
    return df_cat


def prepare_stye_manifest():
    print("\n" + "=" * 60)
    print("2. PREPARING STYE / HORDEOLUM DATASET MANIFEST")
    print("=" * 60)

    stye_dir = Path(os.path.join(WORKSPACE_ROOT, 'stye'))
    normal_dir = Path(os.path.join(
        WORKSPACE_ROOT,
        'Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation',
        'Normal'
    ))

    stye_files = sorted(list(stye_dir.glob('*.jpg')))
    normal_files = sorted(list(normal_dir.glob('*.*')))

    print(f"Found {len(stye_files)} stye images and {len(normal_files)} normal ocular control images.")

    records = []

    # 1. Process Stye images
    for p in stye_files:
        fn = p.name
        if '_jpg.rf.' in fn:
            base_id = fn.split('_jpg.rf.')[0]
        elif '.rf.' in fn:
            base_id = fn.split('.rf.')[0]
        else:
            base_id = fn.split('.')[0]

        rel_path = os.path.relpath(p, WORKSPACE_ROOT).replace('\\', '/')
        md5_val = compute_md5(str(p))
        with Image.open(p) as im:
            w, h = im.size

        records.append({
            'image_id': f"stye_{Path(fn).stem}",
            'filename': fn,
            'relative_path': rel_path,
            'subject_id': f"stye_subj_{base_id}",
            'label': 1,
            'class_name': 'stye_hordeolum',
            'dataset_source': 'Roboflow_Stye_Hordeolum',
            'modality': 'anterior_eyelid_photography',
            'md5': md5_val,
            'width': w,
            'height': h,
            'aspect_ratio': round(w / h, 4)
        })

    # 2. Process Normal images (negative controls)
    for p in normal_files:
        fn = p.name
        base_id = Path(fn).stem
        rel_path = os.path.relpath(p, WORKSPACE_ROOT).replace('\\', '/')
        md5_val = compute_md5(str(p))
        with Image.open(p) as im:
            w, h = im.size

        records.append({
            'image_id': f"normal_{Path(fn).stem}",
            'filename': fn,
            'relative_path': rel_path,
            'subject_id': f"normal_subj_{base_id}",
            'label': 0,
            'class_name': 'normal',
            'dataset_source': 'Eye_Diseases_Classification_Normal_Control',
            'modality': 'anterior_ocular_photography',
            'md5': md5_val,
            'width': w,
            'height': h,
            'aspect_ratio': round(w / h, 4)
        })

    df_stye = pd.DataFrame(records)
    print(f"Total Stye + Control dataset size: {len(df_stye)} images.")
    print(f"Unique subjects: {df_stye['subject_id'].nunique()}")
    print("Class breakdown:\n", df_stye['class_name'].value_counts())

    # Subject-isolated split (70% train, 15% val, 15% test)
    # Stratified at subject level by class label
    np.random.seed(42)
    subj_df = df_stye[['subject_id', 'label']].drop_duplicates()
    subject_to_split = {}

    for cls in [0, 1]:
        cls_subjs = subj_df[subj_df['label'] == cls]['subject_id'].tolist()
        np.random.shuffle(cls_subjs)
        n = len(cls_subjs)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        
        train_s = cls_subjs[:n_train]
        val_s = cls_subjs[n_train:n_train + n_val]
        test_s = cls_subjs[n_train + n_val:]

        for s in train_s:
            subject_to_split[s] = 'train'
        for s in val_s:
            subject_to_split[s] = 'val'
        for s in test_s:
            subject_to_split[s] = 'test'

    df_stye['split'] = df_stye['subject_id'].map(subject_to_split)

    print("\nSubject-Isolated Split Results:")
    print("Subjects per split:", Counter(subject_to_split.values()))
    print("Images per split:\n", df_stye['split'].value_counts())
    print("Split vs Class:\n", pd.crosstab(df_stye['split'], df_stye['class_name']))

    manifest_out = os.path.join(WORKSPACE_ROOT, 'data', 'manifests', 'stye_hordeolum_manifest.csv')
    os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
    df_stye.to_csv(manifest_out, index=False)
    print(f"Saved Stye manifest to: {manifest_out}")
    return df_stye

if __name__ == '__main__':
    prepare_cataract_manifest()
    prepare_stye_manifest()
    print("\n=== DATASET PREPARATION AND LEAK-FREE SPLITTING COMPLETED SUCCESSFULLY ===")
