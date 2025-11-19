

# src/models.py
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from transformers import Wav2Vec2Model


# ------------------------------
# Encoder backbone (Wav2Vec2)
# ------------------------------
class EncoderBackbone(nn.Module):
    """
    Wraps a HuggingFace Wav2Vec2 encoder.
    - Loads weights via safetensors to avoid torch.load() restriction.
    - 'freeze' can lock all encoder params for linear-probe style training.
    - 'unfreeze_top_k(k)' later lets you adapt the last k transformer blocks.
    """
    def __init__(self, backbone_id: str, freeze: bool = True):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(
            backbone_id,
            use_safetensors=True,          # avoid .bin torch.load issues
            local_files_only=False
        )
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _frame_rate_hz(self) -> float:
        """
        The last_hidden_state from Wav2Vec2 is typically ~50 Hz (20ms stride).
        We don't hardcode it; the code below computes fps dynamically per batch
        using lengths_sec and T, so this is informational only.
        """
        return 50.0

    def unfreeze_top_k(self, k: int = 2):
        """
        Unfreeze the last k transformer blocks to adapt to your data.
        Works for Wav2Vec2 (self.encoder.encoder.layers).
        """
        if k <= 0:
            return
        try:
            layers = self.encoder.encoder.layers
        except AttributeError:
            # In case of slightly different encoder attributes in other models
            layers = self.encoder.transformer.encoder.layers
        for p in layers[-k:].parameters():
            p.requires_grad = True

    def forward(self, input_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_values: (B, T_wav) float32 waveforms in [-1,1]
            attention_mask: (B, T_wav) optional mask
        Returns:
            H: (B, T_frames, D) encoder hidden states
        """
        out = self.encoder(input_values=input_values,
                           attention_mask=attention_mask,
                           output_hidden_states=False,
                           return_dict=True)
        return out.last_hidden_state  # (B, T_frames, D)


# ------------------------------
# Utterance-level classifier
# ------------------------------
class UtteranceClassifier(nn.Module):
    """
    Simple utterance-level MLP on top of mean-pooled encoder frames.
    """
    def __init__(self,
                 backbone_id: str,
                 num_classes: int,
                 freeze_backbone: bool = True,
                 hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = EncoderBackbone(backbone_id, freeze_backbone)
        d = self.backbone.encoder.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self,
                input_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns:
            logits: (B, C)
        """
        H = self.backbone(input_values, attention_mask)   # (B, T, D)

        # Masked mean pooling over time (correct + numerically stable)
        if attention_mask is not None:
            # project mask to frame domain if it’s already in frame domain,
            # otherwise we assume processor aligned it; here we assume aligned:
            mask = attention_mask
            if mask.dim() == 2 and mask.size(1) != H.size(1):
                # if not aligned (rare), fallback to simple mean
                pooled = H.mean(dim=1)
            else:
                mask = mask.unsqueeze(-1).float()              # (B, T, 1)
                pooled = (H * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = H.mean(dim=1)

        logits = self.mlp(pooled)  # (B, C)
        return logits


# ------------------------------
# Temporal heads for segmentation
# ------------------------------
class TemporalHeadGRU(nn.Module):
    """
    BiGRU over segment sequence -> per-segment logits.
    """
    def __init__(self,
                 d_in: int,
                 hidden: int = 256,
                 layers: int = 2,
                 dropout: float = 0.1,
                 num_classes: int = 7):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=d_in,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden * 2, num_classes)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S: (B, S_segments, D)
        Returns:
            Z: (B, S_segments, C)
        """
        Z, _ = self.rnn(S)
        return self.proj(Z)


class TemporalHeadTransformer(nn.Module):
    """
    Lightweight Transformer encoder over segments.
    """
    def __init__(self,
                 d_in: int,
                 num_layers: int = 2,
                 nhead: int = 4,
                 dropout: float = 0.1,
                 num_classes: int = 7):
        super().__init__()
        block = nn.TransformerEncoderLayer(
            d_model=d_in,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.enc = nn.TransformerEncoder(block, num_layers=num_layers)
        self.proj = nn.Linear(d_in, num_classes)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S: (B, S_segments, D)
        Returns:
            Z: (B, S_segments, C)
        """
        Z = self.enc(S)
        return self.proj(Z)


# ------------------------------
# Segmental (weakly-supervised) model
# ------------------------------
class SegmentalEmotion(nn.Module):
    """
    End-to-end weakly supervised segmental SER:
      - Wav2Vec2 backbone -> frame features (B, T, D)
      - Frame->Segment pooling with (win_sec, hop_sec)
      - Temporal head (GRU or Transformer) -> per-segment logits (B, S, C)

    forward(...) returns:
      logits_per_segment: (B, S, C)
      segment_mask:       (B, S)  [True for valid segments, False for padding]
    """
    def __init__(self,
                 backbone_id: str,
                 num_classes: int,
                 freeze_backbone: bool = True,
                 head_type: str = 'gru',
                 hidden: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = EncoderBackbone(backbone_id, freeze_backbone)
        d = self.backbone.encoder.config.hidden_size

        if head_type.lower() == 'gru':
            self.head = TemporalHeadGRU(
                d_in=d, hidden=hidden, layers=num_layers,
                dropout=dropout, num_classes=num_classes
            )
        else:
            self.head = TemporalHeadTransformer(
                d_in=d, num_layers=num_layers, nhead=4,
                dropout=dropout, num_classes=num_classes
            )

    @staticmethod
    def seg_from_frames(H: torch.Tensor,
                        lengths_sec: List[float],
                        win_sec: float = 1.0,
                        hop_sec: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert frame sequence (B, T, D) into segment sequence by
        mean-pooling overlapping windows. Handles variable lengths and pads.

        Args:
            H: (B, T, D)
            lengths_sec: list of B floats (per-utterance durations in seconds)
            win_sec: window size in seconds
            hop_sec: hop size in seconds
        Returns:
            S_pad: (B, S_max, D) padded segments
            M:     (B, S_max)    True for valid segments
        """
        device = H.device
        B, T, D = H.shape
        segs_per_batch = []
        masks = []

        for b in range(B):
            dur = float(max(lengths_sec[b], 1e-3))
            # frames-per-second from T and duration
            fps = float(T) / dur
            win = max(1, int(round(fps * float(win_sec))))
            hop = max(1, int(round(fps * float(hop_sec))))

            cur = []
            # slide over frames
            t = 0
            while t + win <= T:
                cur.append(H[b, t:t + win].mean(dim=0))  # (D,)
                t += hop
            if not cur:
                cur = [H[b].mean(dim=0)]  # fallback single segment

            S = torch.stack(cur, dim=0)       # (S_i, D)
            segs_per_batch.append(S)
            masks.append(torch.ones(S.size(0), dtype=torch.bool, device=device))

        S_max = max(x.size(0) for x in segs_per_batch)
        S_pad = torch.zeros(B, S_max, D, device=device)
        M = torch.zeros(B, S_max, dtype=torch.bool, device=device)
        for b, S in enumerate(segs_per_batch):
            n = S.size(0)
            S_pad[b, :n] = S
            M[b, :n] = True
        return S_pad, M

    def forward(self,
                input_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                lengths_sec: Optional[List[float]] = None,
                win_sec: float = 1.0,
                hop_sec: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_values: (B, T_wav)
            attention_mask: (B, T_wav) or None
            lengths_sec: list of floats, length B
            win_sec/hop_sec: segment params in seconds
        Returns:
            Z:    (B, S, C) per-segment logits
            mask: (B, S)    boolean mask
        """
        H = self.backbone(input_values, attention_mask)  # (B, T_frames, D)

        if lengths_sec is None:
            # if not provided, approximate duration from frames and 50 Hz
            B, T, _ = H.shape
            lengths_sec = [float(T) / 50.0] * B

        S, M = self.seg_from_frames(H, lengths_sec, win_sec, hop_sec)  # (B, S, D), (B, S)
        Z = self.head(S)                                               # (B, S, C)
        return Z, M






















# import torch, torch.nn as nn
# from transformers import AutoModel
# import torch.nn as nn
# from transformers import Wav2Vec2Model    
# import math
# from src.losses import FocalLoss
# from src.data import compute_class_weights



# # class EncoderBackbone(nn.Module):
# #     def __init__(self, backbone_id: str, freeze=True):
# #         super().__init__()
# #         self.encoder = AutoModel.from_pretrained(backbone_id)
# #         if freeze:
# #             for p in self.encoder.parameters(): p.requires_grad = False
# #     def forward(self, x, m=None):
# #         out = self.encoder(input_values=x, attention_mask=m)
# #         return out.last_hidden_state
# class EncoderBackbone(nn.Module):
#     def __init__(self, backbone_id: str, freeze=True):
#         super().__init__()
#         # force safetensors to avoid torch.load on .bin
#         self.encoder = Wav2Vec2Model.from_pretrained(
#             backbone_id,
#             use_safetensors=True,
#             local_files_only=False,
#         )
#         if freeze:
#             for p in self.encoder.parameters():
#                 p.requires_grad = False
#         #new line start
#     def unfreeze_top_k(self, k=2):
#         try:
#             layers = self.encoder.encoder.layers  # Wav2Vec2
#         except AttributeError:
#             layers = self.encoder.layers          # some models expose .layers
#         for p in layers[-k:].parameters():
#             p.requires_grad = True
#         #new line end
    

#     def forward(self, x, m=None):
#         out = self.encoder(input_values=x, attention_mask=m)
#         return out.last_hidden_state

# class UtteranceClassifier(nn.Module):
#     def __init__(self, backbone_id, num_classes, freeze_backbone=True, hidden=256, dropout=0.1):
#         super().__init__()
#         self.backbone = EncoderBackbone(backbone_id, freeze_backbone)
#         d = self.backbone.encoder.config.hidden_size
#         self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes))
#     def forward(self, x, m=None):
#         H = self.backbone(x, m)
#         if m is not None:
#             mask = m.unsqueeze(-1).float()
#             H = (H*mask).sum(1)/mask.sum(1).clamp_min(1)
#         else:
#             H = H.mean(1)
#         return self.mlp(H)

# class TemporalHeadGRU(nn.Module):
#     def __init__(self, d, hidden=256, layers=2, dropout=0.1, classes=7):
#         super().__init__()
#         self.rnn = nn.GRU(d, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers>1 else 0.0)
#         self.proj = nn.Linear(hidden*2, classes)
#     def forward(self, S): Z,_=self.rnn(S); return self.proj(Z)

# class SegmentalEmotion(nn.Module):
#     def __init__(self, backbone_id, num_classes, freeze_backbone=True, head_type='gru', hidden=256, layers=2, dropout=0.1):
#         super().__init__()
#         self.backbone = EncoderBackbone(backbone_id, freeze_backbone)
#         d = self.backbone.encoder.config.hidden_size
#         if head_type=='gru':
#             self.head = TemporalHeadGRU(d, hidden, layers, dropout, num_classes)
#         else:
#             enc = nn.TransformerEncoderLayer(d_model=d, nhead=4, dropout=dropout, batch_first=True)
#             self.head = nn.Sequential(nn.TransformerEncoder(enc, num_layers=2), nn.Linear(d, num_classes))
#     @staticmethod
#     def seg_from_frames(H, lengths_sec, win_sec=1.0, hop_sec=0.5):
#         B, T, D = H.shape
#         outs = []; masks = []
#         for b in range(B):
#             # --- NEW: make sure dur is a python float ---
#             val = lengths_sec[b]
#             if isinstance(val, torch.Tensor):
#                 dur = float(val.item())
#             else:
#                 dur = float(val)
#             dur = max(dur, 1e-3)
#             fps = T / dur

#             win = max(1, int(round(fps * win_sec)))
#             hop = max(1, int(round(fps * hop_sec)))

#             segs = []
#             for s in range(0, T - win + 1, hop):
#                 segs.append(H[b, s:s+win].mean(0))
#             if not segs:
#                 segs = [H[b].mean(0)]
#             S = torch.stack(segs, 0)
#             outs.append(S)
#             masks.append(torch.ones(S.shape[0], dtype=torch.bool, device=H.device))

#         maxS = max(x.shape[0] for x in outs)
#         S_pad = torch.zeros(B, maxS, D, device=H.device)
#         M = torch.zeros(B, maxS, dtype=torch.bool, device=H.device)
#         for b, S in enumerate(outs):
#             S_pad[b, :S.shape[0]] = S
#             M[b, :S.shape[0]] = True
#         return S_pad, M
#     # def seg_from_frames(H, lengths_sec, win_sec=1.0, hop_sec=0.5):
#     #     B,T,D = H.shape
#     #     outs=[]; masks=[]
#     #     for b in range(B):
#     #         dur=max(lengths_sec[b],1e-3); fps=T/dur
#     #         win=max(1, int(round(fps*win_sec))); hop=max(1, int(round(fps*hop_sec)))
#     #         segs=[]
#     #         for s in range(0, T-win+1, hop): segs.append(H[b, s:s+win].mean(0))
#     #         if not segs: segs=[H[b].mean(0)]
#     #         S=torch.stack(segs,0); outs.append(S); masks.append(torch.ones(S.shape[0], dtype=torch.bool, device=H.device))
#     #     maxS=max(x.shape[0] for x in outs)
#     #     S_pad=torch.zeros(B,maxS,D,device=H.device); M=torch.zeros(B,maxS,dtype=torch.bool,device=H.device)
#     #     for b,S in enumerate(outs): S_pad[b,:S.shape[0]]=S; M[b,:S.shape[0]]=True
#     #     return S_pad,M
#     def forward(self,x,m=None,lengths_sec=None,win_sec=1.0,hop_sec=0.5):
#         H=self.backbone(x,m)
#         S,M=self.seg_from_frames(H,lengths_sec,win_sec,hop_sec)
#         Z=self.head(S); return Z,M
