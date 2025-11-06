import torch, torch.nn as nn
from transformers import AutoModel
import torch.nn as nn
from transformers import Wav2Vec2Model      


# class EncoderBackbone(nn.Module):
#     def __init__(self, backbone_id: str, freeze=True):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(backbone_id)
#         if freeze:
#             for p in self.encoder.parameters(): p.requires_grad = False
#     def forward(self, x, m=None):
#         out = self.encoder(input_values=x, attention_mask=m)
#         return out.last_hidden_state
class EncoderBackbone(nn.Module):
    def __init__(self, backbone_id: str, freeze=True):
        super().__init__()
        # force safetensors to avoid torch.load on .bin
        self.encoder = Wav2Vec2Model.from_pretrained(
            backbone_id,
            use_safetensors=True,
            local_files_only=False,
        )
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x, m=None):
        out = self.encoder(input_values=x, attention_mask=m)
        return out.last_hidden_state

class UtteranceClassifier(nn.Module):
    def __init__(self, backbone_id, num_classes, freeze_backbone=True, hidden=256, dropout=0.1):
        super().__init__()
        self.backbone = EncoderBackbone(backbone_id, freeze_backbone)
        d = self.backbone.encoder.config.hidden_size
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes))
    def forward(self, x, m=None):
        H = self.backbone(x, m)
        if m is not None:
            mask = m.unsqueeze(-1).float()
            H = (H*mask).sum(1)/mask.sum(1).clamp_min(1)
        else:
            H = H.mean(1)
        return self.mlp(H)

class TemporalHeadGRU(nn.Module):
    def __init__(self, d, hidden=256, layers=2, dropout=0.1, classes=7):
        super().__init__()
        self.rnn = nn.GRU(d, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers>1 else 0.0)
        self.proj = nn.Linear(hidden*2, classes)
    def forward(self, S): Z,_=self.rnn(S); return self.proj(Z)

class SegmentalEmotion(nn.Module):
    def __init__(self, backbone_id, num_classes, freeze_backbone=True, head_type='gru', hidden=256, layers=2, dropout=0.1):
        super().__init__()
        self.backbone = EncoderBackbone(backbone_id, freeze_backbone)
        d = self.backbone.encoder.config.hidden_size
        if head_type=='gru':
            self.head = TemporalHeadGRU(d, hidden, layers, dropout, num_classes)
        else:
            enc = nn.TransformerEncoderLayer(d_model=d, nhead=4, dropout=dropout, batch_first=True)
            self.head = nn.Sequential(nn.TransformerEncoder(enc, num_layers=2), nn.Linear(d, num_classes))
    @staticmethod
    def seg_from_frames(H, lengths_sec, win_sec=1.0, hop_sec=0.5):
        B, T, D = H.shape
        outs = []; masks = []
        for b in range(B):
            # --- NEW: make sure dur is a python float ---
            val = lengths_sec[b]
            if isinstance(val, torch.Tensor):
                dur = float(val.item())
            else:
                dur = float(val)
            dur = max(dur, 1e-3)
            fps = T / dur

            win = max(1, int(round(fps * win_sec)))
            hop = max(1, int(round(fps * hop_sec)))

            segs = []
            for s in range(0, T - win + 1, hop):
                segs.append(H[b, s:s+win].mean(0))
            if not segs:
                segs = [H[b].mean(0)]
            S = torch.stack(segs, 0)
            outs.append(S)
            masks.append(torch.ones(S.shape[0], dtype=torch.bool, device=H.device))

        maxS = max(x.shape[0] for x in outs)
        S_pad = torch.zeros(B, maxS, D, device=H.device)
        M = torch.zeros(B, maxS, dtype=torch.bool, device=H.device)
        for b, S in enumerate(outs):
            S_pad[b, :S.shape[0]] = S
            M[b, :S.shape[0]] = True
        return S_pad, M
    # def seg_from_frames(H, lengths_sec, win_sec=1.0, hop_sec=0.5):
    #     B,T,D = H.shape
    #     outs=[]; masks=[]
    #     for b in range(B):
    #         dur=max(lengths_sec[b],1e-3); fps=T/dur
    #         win=max(1, int(round(fps*win_sec))); hop=max(1, int(round(fps*hop_sec)))
    #         segs=[]
    #         for s in range(0, T-win+1, hop): segs.append(H[b, s:s+win].mean(0))
    #         if not segs: segs=[H[b].mean(0)]
    #         S=torch.stack(segs,0); outs.append(S); masks.append(torch.ones(S.shape[0], dtype=torch.bool, device=H.device))
    #     maxS=max(x.shape[0] for x in outs)
    #     S_pad=torch.zeros(B,maxS,D,device=H.device); M=torch.zeros(B,maxS,dtype=torch.bool,device=H.device)
    #     for b,S in enumerate(outs): S_pad[b,:S.shape[0]]=S; M[b,:S.shape[0]]=True
    #     return S_pad,M
    def forward(self,x,m=None,lengths_sec=None,win_sec=1.0,hop_sec=0.5):
        H=self.backbone(x,m)
        S,M=self.seg_from_frames(H,lengths_sec,win_sec,hop_sec)
        Z=self.head(S); return Z,M
