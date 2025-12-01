# analysis/eval_with_confusion_matrix.py

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from src.data import SpeechDataset, Collator
from src.models import UtteranceClassifier, SegmentalEmotion
from src.metrics import macro_f1, uar


def build_dataloader(cfg, split, label_map):
    ds = SpeechDataset(
        cfg["data"]["manifest_path"],
        split,
        label_map,
        sr=cfg["sample_rate"],
    )
    # no augmentation during eval
    setattr(ds, "augment", False)

    col = Collator(cfg["model"]["backbone_id"], cfg["sample_rate"])
    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=col,
        pin_memory=True,
    )
    return dl


def load_model_and_ckpt(cfg, mode, ckpt_path, device):
    emotions = cfg["data"]["emotions"]
    num_classes = len(emotions)

    if mode == "baseline":
        model = UtteranceClassifier(
            cfg["model"]["backbone_id"],
            num_classes,
            cfg["model"]["freeze_backbone"],
            cfg["model"]["hidden_size"],
            cfg["model"]["dropout"],
        )
    else:  # weakseg
        model = SegmentalEmotion(
            cfg["model"]["backbone_id"],
            num_classes,
            cfg["model"]["freeze_backbone"],
            cfg["model"]["head"],
            cfg["model"]["hidden_size"],
            cfg["model"]["num_layers"],
            cfg["model"]["dropout"],
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def eval_and_collect(cfg, mode, split, ckpt_path, device):
    emotions = cfg["data"]["emotions"]
    label_map = {e: i for i, e in enumerate(emotions)}

    dl = build_dataloader(cfg, split, label_map)
    model = load_model_and_ckpt(cfg, mode, ckpt_path, device)

    ys, yh = [], []

    with torch.no_grad():
        for b in dl:
            x = b["input_values"].to(device)
            m = b.get("attention_mask")
            m = m.to(device) if m is not None else None
            y = b["labels"].to(device)

            if mode == "baseline":
                logits = model(x, m)              # (B, C)
            else:
                lengths = b["lengths_sec"]        # list of floats
                Z, mask = model(
                    x,
                    m,
                    lengths,
                    cfg["segment"]["win_sec"],
                    cfg["segment"]["hop_sec"],
                )
                # bag logits: log-mean-exp over segments
                bag_logits = Z.logsumexp(1) - torch.log(
                    mask.sum(1, keepdim=True).clamp_min(1).float()
                )
                logits = bag_logits               # (B, C)

            pred = logits.argmax(-1)

            ys += y.cpu().tolist()
            yh += pred.cpu().tolist()

    f1 = macro_f1(ys, yh, num_classes=len(emotions))
    r = uar(ys, yh, num_classes=len(emotions))
    print(f"[EVAL][{mode}][{split}] N={len(ys)} unique_true={len(set(ys))} unique_pred={len(set(yh))}")
    print(f"[EVAL][{mode}][{split}] Macro-F1={f1:.4f} UAR={r:.4f}")
    return ys, yh, f1, r


def plot_confusion_matrix(ys, yh, emotions, title, out_path):
    cm = confusion_matrix(ys, yh, labels=list(range(len(emotions))))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(emotions)),
        yticks=range(len(emotions)),
        xticklabels=emotions,
        yticklabels=emotions,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # print numbers
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument("--mode", choices=["baseline", "weakseg"], required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    emotions = cfg["data"]["emotions"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ys, yh, f1, r = eval_and_collect(cfg, args.mode, args.split, args.ckpt, device)

    if args.out is None:
        os.makedirs("analysis", exist_ok=True)
        args.out = f"analysis/confmat_{args.mode}_{args.split}.png"

    title = f"{args.mode.upper()} - {args.split} (F1={f1:.3f}, UAR={r:.3f})"
    plot_confusion_matrix(ys, yh, emotions, title, args.out)


if __name__ == "__main__":
    main()
