# src/train.py

import argparse, yaml, os, math, torch, torch.nn.functional as F
import csv  # NEW: for logging to CSV
from torch.utils.data import DataLoader
from transformers import logging as hf_logging
from tqdm import tqdm

from src.utils import set_seed, ensure_dir
from src.data import SpeechDataset, Collator
from src.models import UtteranceClassifier, SegmentalEmotion
from src.losses import bag_loss, temporal_consistency_loss, change_sparsity_loss
from src.metrics import macro_f1, uar

hf_logging.set_verbosity_error()


def _make_dataloaders(cfg, label_map):
    """Create train/val dataloaders with the correct processor for the backbone."""
    ds_tr = SpeechDataset(cfg['data']['manifest_path'], 'train', label_map, sr=cfg['sample_rate'])
    ds_va = SpeechDataset(cfg['data']['manifest_path'], 'val',   label_map, sr=cfg['sample_rate'])

    # ✅ IMPORTANT: augmentation only for training, never for validation
    # (prevents NaNs and keeps validation distribution stable)
    setattr(ds_tr, "augment", bool(cfg.get('data', {}).get('augment', True)))
    setattr(ds_va, "augment", False)

    col = Collator(cfg['model']['backbone_id'], cfg['sample_rate'])

    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        collate_fn=col,
        pin_memory=True
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        collate_fn=col,
        pin_memory=True
    )
    return dl_tr, dl_va


def _maybe_scheduler(opt, dl_tr_len, cfg):
    """Cosine warmup scheduler (optional via YAML)."""
    if cfg["train"].get("scheduler") == "cosine_warmup":
        total_steps = max(1, dl_tr_len * cfg["train"]["epochs"])
        warmup_steps = int(total_steps * float(cfg["train"].get("warmup_ratio", 0.06)))

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return None


def _maybe_unfreeze_top_k(model, cfg):
    """Unfreeze last K transformer blocks if requested in YAML."""
    k = int(cfg["model"].get("unfreeze_top_k", 0))
    if k > 0 and hasattr(model, "backbone") and hasattr(model.backbone, "unfreeze_top_k"):
        model.backbone.unfreeze_top_k(k)


def train_baseline(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emotions = cfg['data']['emotions']
    label_map = {e: i for i, e in enumerate(emotions)}

    dl_tr, dl_va = _make_dataloaders(cfg, label_map)

    model = UtteranceClassifier(
        cfg['model']['backbone_id'],
        len(emotions),
        cfg['model']['freeze_backbone'],
        cfg['model']['hidden_size'],
        cfg['model']['dropout']
    ).to(device)

    _maybe_unfreeze_top_k(model, cfg)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay']
    )
    scheduler = _maybe_scheduler(opt, len(dl_tr), cfg)

    scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg['train']['amp']))
    ensure_dir(cfg['out_dir'])

    # NEW: set up CSV logging file
    log_dir = os.path.join(cfg['out_dir'], 'logs')
    ensure_dir(log_dir)
    baseline_log_path = os.path.join(log_dir, 'baseline_training.csv')
    with open(baseline_log_path, 'w', newline='') as f_log:
        writer = csv.writer(f_log)
        writer.writerow(['epoch', 'train_loss', 'val_macro_f1', 'val_uar'])

    best = 0.0
    bad_epochs = 0
    patience = int(cfg['train'].get('early_stop_patience', 8))

    for ep in range(cfg['train']['epochs']):
        model.train()
        pbar = tqdm(dl_tr, desc=f"BL {ep+1}/{cfg['train']['epochs']}")

        # NEW: accumulate loss per epoch
        epoch_loss_sum = 0.0
        n_batches = 0

        for i, b in enumerate(pbar):
            x = b['input_values'].to(device)
            m = b.get('attention_mask'); m = m.to(device) if m is not None else None
            y = b['labels'].to(device)

            with torch.amp.autocast('cuda', enabled=bool(cfg['train']['amp'])):
                logits = model(x, m)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            if (i + 1) % cfg['train']['grad_accum'] == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            epoch_loss_sum += float(loss.item())   # NEW
            n_batches += 1                         # NEW

            pbar.set_postfix(loss=float(loss.item()), lr=opt.param_groups[0]['lr'])

        # NEW: mean train loss for this epoch
        mean_train_loss = epoch_loss_sum / max(1, n_batches)

        # ---- validation ----
        model.eval()
        ys, yh = [], []
        with torch.no_grad():
            for b in dl_va:
                x = b['input_values'].to(device)
                m = b.get('attention_mask'); m = m.to(device) if m is not None else None
                y = b['labels'].to(device)
                pred = model(x, m).argmax(-1)
                ys += y.cpu().tolist()
                yh += pred.cpu().tolist()

        print(f"[VAL] N={len(ys)} unique_true={len(set(ys))} unique_pred={len(set(yh))}")
        f1 = macro_f1(ys, yh, num_classes=len(emotions))
        rec = uar(ys, yh, num_classes=len(emotions))
        print(f"Val Macro-F1={f1:.4f} UAR={rec:.4f}")

        # NEW: append epoch stats to CSV log
        with open(baseline_log_path, 'a', newline='') as f_log:
            writer = csv.writer(f_log)
            writer.writerow([ep + 1,
                             f"{mean_train_loss:.6f}",
                             f"{f1:.4f}",
                             f"{rec:.4f}"])

        if f1 > best:
            best = f1
            bad_epochs = 0
            torch.save({'state_dict': model.state_dict(), 'cfg': cfg},
                       os.path.join(cfg['out_dir'], 'baseline_best.pt'))
            print(f"✔ New best Macro-F1={best:.4f} → saved to {os.path.join(cfg['out_dir'],'baseline_best.pt')}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"⛔ Early stopping at epoch {ep+1} (best Macro-F1={best:.4f})")
                break

    # save last
    torch.save({'state_dict': model.state_dict(), 'cfg': cfg},
               os.path.join(cfg['out_dir'], 'baseline_last.pt'))
    print(f"Training ended. Best Macro-F1={best:.4f}. Files: baseline_best.pt / baseline_last.pt")


def train_weakseg(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    emotions = cfg['data']['emotions']
    label_map = {e: i for i, e in enumerate(emotions)}

    dl_tr, dl_va = _make_dataloaders(cfg, label_map)

    model = SegmentalEmotion(
        cfg['model']['backbone_id'],
        len(emotions),
        cfg['model']['freeze_backbone'],
        cfg['model']['head'],
        cfg['model']['hidden_size'],
        cfg['model']['num_layers'],
        cfg['model']['dropout']
    ).to(device)

    _maybe_unfreeze_top_k(model, cfg)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay']
    )
    scheduler = _maybe_scheduler(opt, len(dl_tr), cfg)

    scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg['train']['amp']))
    ensure_dir(cfg['out_dir'])

    # NEW: CSV logging for weakseg
    log_dir = os.path.join(cfg['out_dir'], 'logs')
    ensure_dir(log_dir)
    weakseg_log_path = os.path.join(log_dir, 'weakseg_training.csv')
    with open(weakseg_log_path, 'w', newline='') as f_log:
        writer = csv.writer(f_log)
        writer.writerow(['epoch', 'train_loss', 'val_macro_f1', 'val_uar'])

    best = 0.0
    bad_epochs = 0
    patience = int(cfg['train'].get('early_stop_patience', 8))

    for ep in range(cfg['train']['epochs']):
        model.train()
        pbar = tqdm(dl_tr, desc=f"WS {ep+1}/{cfg['train']['epochs']}")

        # NEW: accumulate epoch loss
        epoch_loss_sum = 0.0
        n_batches = 0

        for i, b in enumerate(pbar):
            x = b['input_values'].to(device)
            m = b.get('attention_mask'); m = m.to(device) if m is not None else None
            y = b['labels'].to(device)
            lengths = b['lengths_sec']  # list of floats from collator

            with torch.amp.autocast('cuda', enabled=bool(cfg['train']['amp'])):
                Z, mask = model(
                    x, m, lengths,
                    cfg['segment']['win_sec'],
                    cfg['segment']['hop_sec']
                )
                Lb = bag_loss(Z, y, mask)
                Lt = temporal_consistency_loss(Z, mask)
                Ls = change_sparsity_loss(Z, mask)
                loss = Lb + cfg['loss']['lambda_temp'] * Lt + cfg['loss']['lambda_sparse'] * Ls

            scaler.scale(loss).backward()
            if (i + 1) % cfg['train']['grad_accum'] == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            epoch_loss_sum += float(loss.item())   # NEW
            n_batches += 1                         # NEW

            pbar.set_postfix(loss=float(loss.item()), lr=opt.param_groups[0]['lr'])

        # NEW: mean loss for this epoch
        mean_train_loss = epoch_loss_sum / max(1, n_batches)

        # ---- validation ----
        model.eval()
        ys, yh = [], []
        with torch.no_grad():
            for b in dl_va:
                x = b['input_values'].to(device)
                m = b.get('attention_mask'); m = m.to(device) if m is not None else None
                y = b['labels'].to(device)
                lengths = b['lengths_sec']  # list of floats
                Z, mask = model(
                    x, m, lengths,
                    cfg['segment']['win_sec'],
                    cfg['segment']['hop_sec']
                )
                # log-mean-exp bag pooling
                bag_logits = (Z.logsumexp(1) - torch.log(mask.sum(1, keepdim=True).clamp_min(1).float()))
                pred = bag_logits.argmax(-1)
                ys += y.cpu().tolist()
                yh += pred.cpu().tolist()

        print(f"[VAL] N={len(ys)} unique_true={len(set(ys))} unique_pred={len(set(yh))}")
        f1 = macro_f1(ys, yh, num_classes=len(emotions))
        rec = uar(ys, yh, num_classes=len(emotions))
        print(f"Val Macro-F1={f1:.4f} UAR={rec:.4f}")

        # NEW: append epoch stats to CSV log
        with open(weakseg_log_path, 'a', newline='') as f_log:
            writer = csv.writer(f_log)
            writer.writerow([ep + 1,
                             f"{mean_train_loss:.6f}",
                             f"{f1:.4f}",
                             f"{rec:.4f}"])

        if f1 > best:
            best = f1
            bad_epochs = 0
            torch.save({'state_dict': model.state_dict(), 'cfg': cfg},
                       os.path.join(cfg['out_dir'], 'weakseg_best.pt'))
            print(f"✔ New best Macro-F1={best:.4f} → saved to {os.path.join(cfg['out_dir'],'weakseg_best.pt')}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"⛔ Early stopping at epoch {ep+1} (best Macro-F1={best:.4f})")
                break

    # save last
    torch.save({'state_dict': model.state_dict(), 'cfg': cfg},
               os.path.join(cfg['out_dir'], 'weakseg_last.pt'))
    print(f"Training ended. Best Macro-F1={best:.4f}. Files: weakseg_best.pt / weakseg_last.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument("--mode", choices=["baseline", "weakseg"], default="weakseg")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, 'r', encoding='utf-8'))
    set_seed(cfg['seed'])

    if a.mode == "baseline":
        train_baseline(cfg)
    else:
        train_weakseg(cfg)
