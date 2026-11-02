#!/usr/bin/env python
"""将 TN3K (MainPatient) 整理成 TinyUSFM 分类训练所需的
data/Cls/tn3k/{train,val,test}/{0,1} 目录结构（软链接，不复制图片）。

用法:
    python prepare_tn3k_cls.py \
        --tn3k_root /home/linux/Downloads/TN3K-2879/TN3K \
        --out_dir ./data/Cls/tn3k
"""
import argparse
import os
from pathlib import Path


def parse_split(txt_path: Path):
    """每行格式: TN3K/<id>/<label>，可能带 '1/' 前缀，自动兼容"""
    samples = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split('/')
            if parts[0] != 'TN3K':
                parts = parts[1:]
            assert len(parts) == 3 and parts[0] == 'TN3K', f'unexpected line: {line!r}'
            img_id, label = parts[1], int(parts[2])
            samples.append((img_id, label))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tn3k_root', type=str,
                        default='/home/linux/Downloads/TN3K-2879/TN3K')
    parser.add_argument('--out_dir', type=str, default='./data/Cls/tn3k')
    args = parser.parse_args()

    tn3k_root = Path(args.tn3k_root).resolve()
    img_dir = tn3k_root / 'img'
    main_patient = tn3k_root / 'MainPatient'
    out_dir = Path(args.out_dir)

    # 部分 split 文件行首多一列 '1'，自动兼容
    splits = {
        'train': parse_split(main_patient / 'train.txt'),
        'val': parse_split(main_patient / 'val.txt'),
        'test': parse_split(main_patient / 'test.txt'),
    }

    for name, samples in splits.items():
        n_created, n_missing = 0, 0
        for img_id, label in samples:
            src = img_dir / f'{img_id}.jpg'
            if not src.exists():
                n_missing += 1
                continue
            dst_dir = out_dir / name / str(label)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
            n_created += 1
        print(f'{name}: linked {n_created}, missing {n_missing}')


if __name__ == '__main__':
    main()
