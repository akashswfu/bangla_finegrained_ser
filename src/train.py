import argparse, yaml, os, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import logging as hf_logging
from src.utils import set_seed, ensure_dir
from src.data import SpeechDataset, Collator
from src.models import UtteranceClassifier, SegmentalEmotion
from src.losses import bag_loss, temporal_consistency_loss, change_sparsity_loss
from src.metrics import macro_f1, uar
from tqdm import tqdm
hf_logging.set_verbosity_error()

def train_baseline(cfg):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    emotions=cfg['data']['emotions']; label_map={e:i for i,e in enumerate(emotions)}
    ds_tr=SpeechDataset(cfg['data']['manifest_path'],'train',label_map,sr=cfg['sample_rate'])
    ds_va=SpeechDataset(cfg['data']['manifest_path'],'val',label_map,sr=cfg['sample_rate'])
    col=Collator(cfg['model']['backbone_id'], cfg['sample_rate'])
    dl_tr=DataLoader(ds_tr,batch_size=cfg['train']['batch_size'],shuffle=True,num_workers=cfg['data']['num_workers'],collate_fn=col)
    dl_va=DataLoader(ds_va,batch_size=cfg['train']['batch_size'],shuffle=False,num_workers=cfg['data']['num_workers'],collate_fn=col)
    model=UtteranceClassifier(cfg['model']['backbone_id'],len(emotions),cfg['model']['freeze_backbone'], cfg['model']['hidden_size'], cfg['model']['dropout']).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    scaler = torch.amp.GradScaler(device_type='cuda', enabled=bool(cfg['train']['amp'])); best=0.0; ensure_dir(cfg['out_dir'])
    for ep in range(cfg['train']['epochs']):
        model.train(); pbar=tqdm(dl_tr, desc=f"BL {ep+1}/{cfg['train']['epochs']}")
        for i,b in enumerate(pbar):
            x=b['input_values'].to(device); m=b.get('attention_mask'); m=m.to(device) if m is not None else None; y=b['labels'].to(device)
            with torch.amp.autocast(device_type='cuda', enabled=bool(cfg['train']['amp'])):
                logits=model(x,m); loss=F.cross_entropy(logits,y)
            scaler.scale(loss).backward()
            if (i+1)%cfg['train']['grad_accum']==0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=float(loss.item()))
        model.eval(); ys=[]; yh=[]
        with torch.no_grad():
            for b in dl_va:
                x=b['input_values'].to(device); m=b.get('attention_mask'); m=m.to(device) if m is not None else None; y=b['labels'].to(device)
                pred=model(x,m).argmax(-1); ys+=y.cpu().tolist(); yh+=pred.cpu().tolist()
        f1=macro_f1(ys,yh); rec=uar(ys,yh); print(f"Val Macro-F1={f1:.4f} UAR={rec:.4f}")
        if f1>best: best=f1; torch.save({'state_dict':model.state_dict(),'cfg':cfg}, os.path.join(cfg['out_dir'],'baseline_best.pt'))

def train_weakseg(cfg):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    emotions=cfg['data']['emotions']; label_map={e:i for i,e in enumerate(emotions)}
    ds_tr=SpeechDataset(cfg['data']['manifest_path'],'train',label_map,sr=cfg['sample_rate'])
    ds_va=SpeechDataset(cfg['data']['manifest_path'],'val',label_map,sr=cfg['sample_rate'])
    col=Collator(cfg['model']['backbone_id'], cfg['sample_rate'])
    dl_tr=DataLoader(ds_tr,batch_size=cfg['train']['batch_size'],shuffle=True,num_workers=cfg['data']['num_workers'],collate_fn=col)
    dl_va=DataLoader(ds_va,batch_size=cfg['train']['batch_size'],shuffle=False,num_workers=cfg['data']['num_workers'],collate_fn=col)
    model=SegmentalEmotion(cfg['model']['backbone_id'],len(emotions),cfg['model']['freeze_backbone'], cfg['model']['head'], cfg['model']['hidden_size'], cfg['model']['num_layers'], cfg['model']['dropout']).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    scaler=torch.cuda.amp.GradScaler(enabled=bool(cfg['train']['amp'])); best=0.0; ensure_dir(cfg['out_dir'])
    for ep in range(cfg['train']['epochs']):
        model.train(); pbar=tqdm(dl_tr, desc=f"WS {ep+1}/{cfg['train']['epochs']}")
        for i,b in enumerate(pbar):
            x=b['input_values'].to(device); m=b.get('attention_mask'); m=m.to(device) if m is not None else None; y=b['labels'].to(device); lengths=b['lengths_sec']
            with torch.cuda.amp.autocast(enabled=bool(cfg['train']['amp'])):
                Z,mask=model(x,m,lengths,cfg['segment']['win_sec'],cfg['segment']['hop_sec'])
                Lb=bag_loss(Z,y,mask); Lt=temporal_consistency_loss(Z,mask); Ls=change_sparsity_loss(Z,mask)
                loss=Lb + cfg['loss']['lambda_temp']*Lt + cfg['loss']['lambda_sparse']*Ls
            scaler.scale(loss).backward()
            if (i+1)%cfg['train']['grad_accum']==0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=float(loss.item()))
        model.eval(); ys=[]; yh=[]
        with torch.no_grad():
            for b in dl_va:
                x=b['input_values'].to(device); m=b.get('attention_mask'); m=m.to(device) if m is not None else None; y=b['labels'].to(device); lengths=b['lengths_sec']
                Z,mask=model(x,m,lengths,cfg['segment']['win_sec'],cfg['segment']['hop_sec'])
                bag_logits=(Z.logsumexp(1) - torch.log(mask.sum(1,keepdim=True).clamp_min(1).float()))
                pred=bag_logits.argmax(-1); ys+=y.cpu().tolist(); yh+=pred.cpu().tolist()
        f1=macro_f1(ys,yh); rec=uar(ys,yh); print(f"Val Macro-F1={f1:.4f} UAR={rec:.4f}")
        if f1>best: best=f1; torch.save({'state_dict':model.state_dict(),'cfg':cfg}, os.path.join(cfg['out_dir'],'weakseg_best.pt'))

if __name__ == "__main__":
    import sys, yaml
    ap=argparse.ArgumentParser(); ap.add_argument("--config", type=str, default="configs/base.yaml"); ap.add_argument("--mode", choices=["baseline","weakseg"], default="weakseg")
    a=ap.parse_args(); cfg=yaml.safe_load(open(a.config,'r',encoding='utf-8'))
    from src.utils import set_seed; set_seed(cfg['seed'])
    if a.mode=="baseline": train_baseline(cfg)
    else: train_weakseg(cfg)
