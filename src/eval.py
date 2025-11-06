import argparse, yaml, torch, os
from torch.utils.data import DataLoader
from src.data import SpeechDataset, Collator
from src.models import UtteranceClassifier
from src.metrics import macro_f1, uar

def main(cfg, split):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    emotions=cfg['data']['emotions']; label_map={e:i for i,e in enumerate(emotions)}
    ds=SpeechDataset(cfg['data']['manifest_path'], split, label_map, sr=cfg['sample_rate'])
    col=Collator(cfg['model']['backbone_id'], cfg['sample_rate'])
    dl=DataLoader(ds,batch_size=cfg['train']['batch_size'],shuffle=False,num_workers=cfg['data']['num_workers'],collate_fn=col)
    model=UtteranceClassifier(cfg['model']['backbone_id'],len(emotions),cfg['model']['freeze_backbone'], cfg['model']['hidden_size'], cfg['model']['dropout']).to(device)
    ys=[]; yh=[]; model.eval()
    with torch.no_grad():
        for b in dl:
            x=b['input_values'].to(device); m=b.get('attention_mask'); m=m.to(device) if m is not None else None; y=b['labels'].to(device)
            pred=model(x,m).argmax(-1); ys+=y.cpu().tolist(); yh+=pred.cpu().tolist()
    print(f"{split} Macro-F1={macro_f1(ys,yh):.4f} UAR={uar(ys,yh):.4f}")

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config", default="configs/base.yaml"); ap.add_argument("--split", default="val")
    a=ap.parse_args(); cfg=yaml.safe_load(open(a.config,'r',encoding='utf-8')); main(cfg, a.split)
