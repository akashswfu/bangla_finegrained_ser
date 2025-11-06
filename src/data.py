# src/data.py

import csv
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor
import librosa


def _safe_mono_float32(wav: np.ndarray) -> np.ndarray:
    """Ensure mono, 1-D, float32, finite, non-empty."""
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()

    wav = np.asarray(wav)

    # stereo -> mono
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    # force 1-D
    wav = wav.reshape(-1)

    # dtype + finite
    wav = wav.astype(np.float32, copy=False)
    if wav.size == 0 or not np.isfinite(wav).all():
        # 10 ms silence fallback
        wav = np.zeros(160, dtype=np.float32)  # assuming 16kHz default; collator will fix again if needed
    return wav


class SpeechDataset(Dataset):
    """
    Reads manifest CSV with columns: path,label,speaker,split
    Example:
      data/wavs/M_10_EVAN_S_9_NEUTRAL_5.wav,neutral,evan,train
    """

    def __init__(self, manifest, split, label_map, sr=16000, min_dur=0.5, max_dur=20.0):
        self.sr = sr
        self.label_map = label_map
        self.items = []

        with open(manifest, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, fieldnames=None)
            # If the CSV has no header row, DictReader will fail.
            # So we detect header based on the first row having keys.
            # But most likely your file has header row: path,label,speaker,split
            header = reader.fieldnames
            if header is None or not {"path", "label", "split"}.issubset(set(header)):
                # Try to read with explicit fieldnames
                f.seek(0)
                reader = csv.DictReader(
                    f, fieldnames=["path", "label", "speaker", "split"]
                )
            for r in reader:
                if r.get("split", "").strip().lower() != split:
                    continue
                path = r.get("path")
                lab_name = r.get("label")
                if path is None or lab_name is None:
                    continue
                if lab_name not in self.label_map:
                    continue
                lab = self.label_map[lab_name]

                # duration check using audio metadata
                try:
                    info = sf.info(path)
                    dur_meta = float(info.frames) / float(info.samplerate)
                except Exception:
                    continue

                if dur_meta < min_dur or dur_meta > max_dur:
                    continue

                # Keep (path, label); duration will be recomputed after resample for accuracy
                self.items.append((path, lab))

        # Optional: sort for stable batches (not required)
        # self.items.sort(key=lambda x: x[0])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, lab = self.items[i]

        # read wav
        wav, sr = sf.read(path, dtype="float32", always_2d=False)

        # sanitize to mono/float32
        wav = _safe_mono_float32(wav)

        # resample if needed
        if sr != self.sr:
            try:
                wav = librosa.resample(y=wav, orig_sr=sr, target_sr=self.sr)
            except TypeError:
                # fallback for older librosa signatures
                wav = librosa.resample(wav, sr, self.sr)

            wav = _safe_mono_float32(wav)

        # final clip (safety)
        wav = np.clip(wav, -1.0, 1.0).astype(np.float32, copy=False)

        # recompute duration after resample
        dur = float(len(wav)) / float(self.sr)

        # return numpy here; collator will convert to tensors
        return {
            "wav": wav,             # np.ndarray (T,)
            "label": int(lab),      # python int
            "dur": dur,             # float
            "path": path,           # optional for debugging
        }


class Collator:
    """
    Collates variable-length waveforms for Wav2Vec2Processor.
    Ensures each item is 1-D mono float32 numpy array.
    """

    def __init__(self, backbone_id: str, sample_rate: int = 16000):
        self.processor = AutoProcessor.from_pretrained(backbone_id)
        self.sample_rate = int(sample_rate)

    def __call__(self, batch):
        # unpack
        wavs = [b["wav"] for b in batch]
        labels = [b["label"] for b in batch]
        durs = [b["dur"] for b in batch]

        # sanitize every wav to mono, 1-D, float32
        wavs_clean = []
        for w in wavs:
            w = _safe_mono_float32(w)
            if w.size == 0 or not np.isfinite(w).all():
                w = np.zeros(int(0.01 * self.sample_rate), dtype=np.float32)
            wavs_clean.append(w)

        # processor will pad to max length, returning torch tensors
        X = self.processor(
            wavs_clean,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        input_values = X["input_values"]  # (B, T)
        attention_mask = X.get(
            "attention_mask", torch.ones_like(input_values, dtype=torch.long)
        )

        labels = torch.tensor(labels, dtype=torch.long)
        # lengths = torch.tensor(durs, dtype=torch.float32) just remove this line

        return {
            "input_values": input_values,        # (B, T) float32
            "attention_mask": attention_mask,    # (B, T) long
            "labels": labels,                    # (B,)   long
            # "lengths_sec": lengths,              # (B,)   float32
            "lengths_sec": durs,                 # (B,)   float list
        }




# import csv, torch, soundfile as sf, numpy as np
# from torch.utils.data import Dataset
# from transformers import AutoProcessor
# import librosa


# class SpeechDataset(Dataset):
#     def __init__(self, manifest, split, label_map, sr=16000, min_dur=0.5, max_dur=20.0):
#         self.sr = sr; self.label_map = label_map; self.items = []
#         with open(manifest, newline='', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             for r in reader:
#                 if r['split'] != split: continue
#                 path = r['path']; lab = self.label_map[r['label']]
#                 try:
#                     info = sf.info(path); dur = info.frames / info.samplerate
#                 except:
#                     continue
#                 if dur < min_dur or dur > max_dur: continue
#                 self.items.append((path, lab, dur))

#     def __len__(self): return len(self.items)
#     def __getitem__(self, i):
#         path, lab, dur = self.items[i]
#         wav, sr = sf.read(path, dtype='float32', always_2d=False)
#         if wav.ndim == 2: wav = wav.mean(axis=1)
#         if sr != self.sr:
#             # import librosa; wav = librosa.resample(wav, sr, self.sr)

#             try:
#                 wav = librosa.resample(y=wav, orig_sr=sr, target_sr=self.sr)
#             except TypeError:
#                 # fallback for older librosa
#                 wav = librosa.resample(wav, sr, self.sr)

#         wav = np.clip(wav, -1, 1).astype('float32')
#         return {'wav': torch.from_numpy(wav), 'label': torch.tensor(lab), 'dur': float(dur)}

# class Collator:
#     def __init__(self, backbone_id: str, sample_rate: int = 16000):
#         self.processor = AutoProcessor.from_pretrained(backbone_id); self.sample_rate = sample_rate
#     def __call__(self, batch):
#         wavs = [b['wav'] for b in batch]; labels = torch.stack([b['label'] for b in batch]); durs = [b['dur'] for b in batch]
#         X = self.processor(wavs, sampling_rate=self.sample_rate, return_tensors='pt', padding=True)
#         out = {'input_values': X['input_values'], 'labels': labels, 'lengths_sec': durs}
#         if 'attention_mask' in X: out['attention_mask'] = X['attention_mask']
#         return out
