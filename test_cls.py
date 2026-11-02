"""在 TN3K 等分类数据集的 test  split 上评估已训练好的 TinyUSFM/USFM 模型。

用法:
    python test_cls.py \
        --checkpoint_weights ./logs/cls/tn3k/TinyUSFM_pretrained/<time>/models/epochXXX_val_X.XXXX.pth \
        --data_dir ./data/Cls/tn3k

若不传 --checkpoint_weights，则自动选取 logs/cls/<dataset>/ 下 val 精度最高的模型。
"""
import torch
import torch.nn as nn
import argparse
import os
import glob
import re
import numpy as np
from datetime import datetime

from utils.evaluate import Evaluator
from utils.data_processing import DataProcessor
from utils.logger import setup_logger
from utils.load_model import load_model


def parse_args():
    parser = argparse.ArgumentParser(description='test hyperparameters')
    parser.add_argument('--model_name', type=str, default='TinyUSFM', help='name of the model')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--data_dir', type=str, default='./data/Cls/tn3k', help='data directory')
    parser.add_argument('--img_size', type=int, default=224, help='image size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='load foundation pretrained weights before overlaying trained weights')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/TinyUSFM.pth',
                        help='foundation pretraining checkpoint (base weights)')
    parser.add_argument('--weights', type=str, default='',
                        help='trained classification model (.pth state_dict); if empty, auto-pick best')
    parser.add_argument('--save_dir', type=str, default='',
                        help='directory to save test results; default ./results/cls/<dataset>/<time>')
    args = parser.parse_args()
    return args


def find_best_weights(args):
    dataset_name = os.path.basename(args.data_dir.rstrip('/'))
    pattern = f"./logs/cls/{dataset_name}/{args.model_name}_pretrained/*/models/*_val_*.pth"
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f'No trained weights found under {pattern}. '
                                f'Train first or pass --weights explicitly.')
    def val_acc(path):
        m = re.search(r'val_([\d.]+)\.pth', os.path.basename(path))
        return float(m.group(1)) if m else -1.0
    best = max(candidates, key=val_acc)
    print(f'Auto-selected best weights: {best}')
    return best


def main(args):
    if not args.weights:
        args.weights = find_best_weights(args)

    dataset_name = os.path.basename(args.data_dir.rstrip('/'))
    if not args.save_dir:
        args.save_dir = os.path.join(f"./results/cls/{dataset_name}",
                                     datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.save_dir, exist_ok=True)

    logger = setup_logger(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading trained weights from: {args.weights}")

    # 复用 load_model 构造同结构模型，再覆盖训练后的权重
    model = load_model(args, device)
    state_dict = torch.load(args.weights, map_location=device)
    load_info = model.load_state_dict(state_dict, strict=True)
    logger.info(f"Trained weights loaded: {load_info}")

    _, _, test_loader = DataProcessor.get_data_loaders(args)
    # 补充文件名（save_predictions 需要）
    image_names = [s[0] for s in test_loader.dataset.samples]

    evaluator = Evaluator()
    metrics, preds, labels = evaluator.evaluate_model(model, test_loader, device)
    evaluator.print_metrics(metrics, phase='test')
    logger.info(f"Test Metrics: {metrics}")

    evaluator.generate_classification_report(labels, preds, args.save_dir)
    evaluator.save_predictions(image_names, preds, labels,
                               os.path.join(args.save_dir, 'test_predictions.xlsx'))

    with open(os.path.join(args.save_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"weights: {args.weights}\ndata_dir: {args.data_dir}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    logger.info(f"Results saved to: {args.save_dir}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
