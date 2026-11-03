"""在 TN3K 等数据集的 test split 上评估已训练好的 TinyUSFM_Seg/USFM_Seg 分割模型。

用法:
    python test_seg.py --data_dir ./data/Seg/tn3k --num_classes 2

若不传 --weights，则自动选取 logs/seg/<dataset>/ 下 Dice 最高的模型。
"""
import torch
import argparse
import os
import glob
import re
from datetime import datetime

from utils.evaluate import Evaluator_seg
from utils.data_processing_seg import DataProcessor
from utils.logger import setup_logger
from utils.load_model_seg import load_model_seg


def parse_args():
    parser = argparse.ArgumentParser(description='seg test hyperparameters')
    parser.add_argument('--model_name', type=str, default='TinyUSFM_Seg')
    parser.add_argument('--data_dir', type=str, default='./data/Seg/tn3k')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--task_type', type=str, default='tumor')
    parser.add_argument('--pretrained', type=str, default='True')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/TinyUSFM.pth',
                        help='foundation pretraining checkpoint (base weights)')
    parser.add_argument('--weights', type=str, default='',
                        help='trained seg model (.pth state_dict); if empty, auto-pick best-Dice')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--save_visual', type=str, default='False',
                        help='save per-image visualization (image/pred/gt)')
    args = parser.parse_args()
    return args


def find_best_weights(args):
    dataset_name = os.path.basename(args.data_dir.rstrip('/'))
    pattern = f"./logs/seg/{dataset_name}/{args.model_name}_pretrained/*/models/*_dice*.pth"
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f'No trained seg weights found under {pattern}. '
                                f'Train first or pass --weights explicitly.')

    def dice(path):
        m = re.search(r'dice([\d.]+)\.pth', os.path.basename(path))
        return float(m.group(1)) if m else -1.0
    best = max(candidates, key=dice)
    print(f'Auto-selected best seg weights: {best}')
    return best


def main(args):
    if not args.weights:
        args.weights = find_best_weights(args)

    dataset_name = os.path.basename(args.data_dir.rstrip('/'))
    if not args.save_dir:
        args.save_dir = os.path.join(f"./results/seg/{dataset_name}",
                                     datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.save_dir, exist_ok=True)

    logger = setup_logger(args.save_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Loading trained seg weights from: {args.weights}")

    model = load_model_seg(args, device)
    state_dict = torch.load(args.weights, map_location=device)
    load_info = model.load_state_dict(state_dict, strict=True)
    logger.info(f"Trained weights loaded: {load_info}")

    _, _, test_loader = DataProcessor.get_data_loaders(args)

    evaluator = Evaluator_seg()
    metrics = evaluator.evaluate_model(model, test_loader, device, args.num_classes)
    evaluator.print_metrics(metrics, phase='test')
    logger.info(f"Test Metrics: {metrics}")

    with open(os.path.join(args.save_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"weights: {args.weights}\ndata_dir: {args.data_dir}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")

    if args.save_visual == 'True':
        import torch as _t
        model.eval()
        images, preds, gts, names = [], [], [], []
        with _t.no_grad():
            for img, mask in test_loader:
                out = model(img.to(device))
                if out.shape[1] == 1:
                    p = (torch.sigmoid(out) > 0.5).float()
                else:
                    p = out.argmax(dim=1, keepdim=True).float()
                images.append(img.cpu()); preds.append(p.cpu()); gts.append(mask.unsqueeze(1).cpu())
                names.extend(test_loader.dataset.image_files)
        evaluator.save_visual_results(_t.cat(images), _t.cat(preds), _t.cat(gts),
                                      os.path.join(args.save_dir, 'visual'), names)
        logger.info(f"Visual results saved to {os.path.join(args.save_dir, 'visual')}")

    logger.info(f"Results saved to: {args.save_dir}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
