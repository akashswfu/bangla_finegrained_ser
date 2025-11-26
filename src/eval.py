# src/eval.py

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader

from src.data import SpeechDataset, Collator
from src.models import UtteranceClassifier, SegmentalEmotion
from src.metrics import macro_f1, uar


def eval_baseline(cfg, ckpt_path, split):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emotions = cfg['data']['emotions']
    label_map = {e: i for i, e in enumerate(emotions)}

    ds = SpeechDataset(
        cfg['data']['manifest_path'],
        split,
        label_map,
        sr=cfg['sample_rate']
    )
    col = Collator(cfg['model']['backbone_id'], cfg['sample_rate'])
    dl = DataLoader(
        ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        collate_fn=col
    )

    # model init
    model = UtteranceClassifier(
        cfg['model']['backbone_id'],
        num_classes=len(emotions),
        freeze_backbone=cfg['model']['freeze_backbone'],
        hidden=cfg['model']['hidden_size'],
        dropout=cfg['model']['dropout']
    ).to(device)

    # load weights
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    ys, yh = [], []
    with torch.no_grad():
        for b in dl:
            x = b['input_values'].to(device)
            m = b.get('attention_mask')
            m = m.to(device) if m is not None else None
            y = b['labels'].to(device)
            pred = model(x, m).argmax(-1)
            ys += y.cpu().tolist()
            yh += pred.cpu().tolist()

    print(f"[EVAL][baseline][{split}] N={len(ys)} "
          f"unique_true={len(set(ys))} unique_pred={len(set(yh))}")
    f1 = macro_f1(ys, yh, num_classes=len(emotions))
    rec = uar(ys, yh, num_classes=len(emotions))
    print(f"[EVAL][baseline][{split}] Macro-F1={f1:.4f} UAR={rec:.4f}")


def eval_weakseg(cfg, ckpt_path, split):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emotions = cfg['data']['emotions']
    label_map = {e: i for i, e in enumerate(emotions)}

    ds = SpeechDataset(
        cfg['data']['manifest_path'],
        split,
        label_map,
        sr=cfg['sample_rate']
    )
    col = Collator(cfg['model']['backbone_id'], cfg['sample_rate'])
    dl = DataLoader(
        ds,
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        collate_fn=col
    )

    model = SegmentalEmotion(
        cfg['model']['backbone_id'],
        num_classes=len(emotions),
        freeze_backbone=cfg['model']['freeze_backbone'],
        head_type=cfg['model']['head'],
        hidden=cfg['model']['hidden_size'],
        num_layers=cfg['model']['num_layers'],
        dropout=cfg['model']['dropout']
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    ys, yh = [], []
    with torch.no_grad():
        for b in dl:
            x = b['input_values'].to(device)
            m = b.get('attention_mask')
            m = m.to(device) if m is not None else None
            y = b['labels'].to(device)
            lengths = b['lengths_sec']   # list of floats
            Z, mask = model(
                x, m, lengths,
                cfg['segment']['win_sec'],
                cfg['segment']['hop_sec']
            )
            # log-mean-exp bag pooling
            bag_logits = Z.logsumexp(1) - torch.log(
                mask.sum(1, keepdim=True).clamp_min(1).float()
            )
            pred = bag_logits.argmax(-1)
            ys += y.cpu().tolist()
            yh += pred.cpu().tolist()

    print(f"[EVAL][weakseg][{split}] N={len(ys)} "
          f"unique_true={len(set(ys))} unique_pred={len(set(yh))}")
    f1 = macro_f1(ys, yh, num_classes=len(emotions))
    rec = uar(ys, yh, num_classes=len(emotions))
    print(f"[EVAL][weakseg][{split}] Macro-F1={f1:.4f} UAR={rec:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to .pt checkpoint (baseline_best.pt / weakseg_best.pt)")
    ap.add_argument("--mode", choices=["baseline", "weakseg"], default="baseline")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    if args.mode == "baseline":
        eval_baseline(cfg, args.ckpt, args.split)
    else:
        eval_weakseg(cfg, args.ckpt, args.split)
