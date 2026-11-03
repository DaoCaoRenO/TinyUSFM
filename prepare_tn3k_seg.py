#!/usr/bin/env python
"""将 TN3K (MainPatient) 整理成 TinyUSFM 分割训练所需的
data/Seg/tn3k/{train,val,test}/{img,label} 目录结构（软链接，不复制）。

img/<id>.jpg 与 label/<id>.jpg 同名配对，label 为二值结节 mask。

用法:
    python prepare_tn3k_seg.py \
        --tn3k_root /home/linux/Downloads/TN3K-2879/TN3K \
        --out_dir ./data/Seg/tn3k
"""
import argparse
from pathlib import Path


def parse_split(txt_path: Path):
    """每行格式: TN3K/<id>/<label>，可能带 '1/' 前缀，自动兼容"""
    ids = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split('/')
            if parts[0] != 'TN3K':
                parts = parts[1:]
            assert len(parts) == 3 and parts[0] == 'TN3K', f'unexpected line: {line!r}'
            ids.append(parts[1])
    return ids


def link_pair(split_dir: Path, src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tn3k_root', type=str,
                        default='/home/linux/Downloads/TN3K-2879/TN3K')
    parser.add_argument('--out_dir', type=str, default='./data/Seg/tn3k')
    args = parser.parse_args()

    tn3k_root = Path(args.tn3k_root).resolve()
    img_root = tn3k_root / 'img'
    label_root = tn3k_root / 'label'
    mp = tn3k_root / 'MainPatient'
    out_dir = Path(args.out_dir)

    splits = {
        'train': parse_split(mp / 'train.txt'),
        'val': parse_split(mp / 'val.txt'),
        'test': parse_split(mp / 'test.txt'),
    }

    for name, ids in splits.items():
        n_ok, n_missing = 0, 0
        for img_id in ids:
            img_src = img_root / f'{img_id}.jpg'
            label_src = label_root / f'{img_id}.jpg'
            if not (img_src.exists() and label_src.exists()):
                n_missing += 1
                continue
            link_pair(name, img_src, out_dir / name / 'img')
            link_pair(name, label_src, out_dir / name / 'label')
            n_ok += 1
        print(f'{name}: paired {n_ok}, missing {n_missing}')


if __name__ == '__main__':
    main()
