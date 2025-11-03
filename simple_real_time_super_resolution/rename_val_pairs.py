#!/usr/bin/env python3
# save as rename_val_pairs.py and run: python3 rename_val_pairs.py

import os, glob, shutil

base = "dataset/SR/RLSR/val_phase_X2"
lr_dir = os.path.join(base, "val_phase_LR")
hr_dir = os.path.join(base, "val_phase_HR")

# get sorted lists
lr_files = sorted(glob.glob(os.path.join(lr_dir, "*")))
hr_files = sorted(glob.glob(os.path.join(hr_dir, "*")))

if len(lr_files) != len(hr_files):
    print("Warning: LR and HR counts differ:", len(lr_files), len(hr_files))

# rename via temp names to avoid collisions
tmp_lr = []
tmp_hr = []
for i, (l, h) in enumerate(zip(lr_files, hr_files)):
    ext_l = os.path.splitext(l)[1]
    ext_h = os.path.splitext(h)[1]
    tmp_l = os.path.join(lr_dir, f".tmp_lr_{i}{ext_l}")
    tmp_h = os.path.join(hr_dir, f".tmp_hr_{i}{ext_h}")
    os.rename(l, tmp_l)
    os.rename(h, tmp_h)
    tmp_lr.append(tmp_l)
    tmp_hr.append(tmp_h)

# now final rename to 0.png ... preserving extension
for i, (tl, th) in enumerate(zip(tmp_lr, tmp_hr)):
    new_l = os.path.join(lr_dir, f"{i}{os.path.splitext(tl)[1]}")
    new_h = os.path.join(hr_dir, f"{i}{os.path.splitext(th)[1]}")
    os.rename(tl, new_l)
    os.rename(th, new_h)

print("Renamed", len(tmp_lr), "pairs to numeric filenames.")

