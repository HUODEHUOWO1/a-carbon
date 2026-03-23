#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import pandas as pd

def load_mapping(mapping_path: Path):
    rows = []
    with mapping_path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            wnid, desc = line.split(' ', 1)
            rows.append((idx, wnid, desc))
    return rows


def main():
    ap = argparse.ArgumentParser(description='Generate formal ImageNet-1k calibration/profile CSVs from a local val directory organized as val/<wnid>/*.JPEG')
    ap.add_argument('--imagenet-val-root', required=True, help='Path to local ImageNet-1k val root organized by wnid directories')
    ap.add_argument('--mapping', default='LOC_synset_mapping.txt', help='Path to LOC_synset_mapping.txt')
    ap.add_argument('--out-dir', required=True, help='Output directory')
    ap.add_argument('--seed', type=int, default=20260317)
    ap.add_argument('--cal-per-class', type=int, default=1, help='Unique calibration images per class')
    ap.add_argument('--profile-per-class', type=int, default=10, help='Unique profiling images per class')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    val_root = Path(args.imagenet_val_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping(Path(args.mapping))

    cal_rows, prof_rows = [], []
    for idx, wnid, desc in mapping:
        cls_dir = val_root / wnid
        if not cls_dir.is_dir():
            raise FileNotFoundError(f'Missing class directory: {cls_dir}')
        imgs = sorted([p for p in cls_dir.iterdir() if p.is_file()])
        need = args.cal_per_class + args.profile_per_class
        if len(imgs) < need:
            raise ValueError(f'{wnid} has {len(imgs)} files, need at least {need}')
        picks = imgs.copy()
        rng.shuffle(picks)
        cal = picks[:args.cal_per_class]
        prof = picks[args.cal_per_class:args.cal_per_class + args.profile_per_class]
        for j, p in enumerate(cal):
            cal_rows.append({
                'sample_id': f'imagenet1k_cal_{idx:04d}_{j:02d}',
                'label': idx,
                'group_id': idx,
                'wnid': wnid,
                'class_name': desc,
                'image_path': str(p),
                'split': 'calibration',
                'paper_use': True,
                'dataset_variant': 'imagenet1k_val_formal',
            })
        for j, p in enumerate(prof):
            prof_rows.append({
                'sample_id': f'imagenet1k_prof_{idx:04d}_{j:02d}',
                'label': idx,
                'group_id': idx,
                'wnid': wnid,
                'class_name': desc,
                'image_path': str(p),
                'split': 'profile',
                'paper_use': True,
                'dataset_variant': 'imagenet1k_val_formal',
            })

    pd.DataFrame(cal_rows).to_csv(out_dir / f'imagenet1k_val_calibration_{len(cal_rows)}.csv', index=False)
    pd.DataFrame(prof_rows).to_csv(out_dir / f'imagenet1k_val_profile_{len(prof_rows)}.csv', index=False)
    pd.DataFrame(mapping, columns=['label','wnid','class_name']).to_csv(out_dir / 'imagenet1k_label_map.csv', index=False)

if __name__ == '__main__':
    main()
