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
        # 10 ms silence fallback (assuming 16kHz default; collator pads anyway)
        wav = np.zeros(160, dtype=np.float32)
    return wav


def compute_class_weights(manifest_path, label_map):
    import collections
    cnt = collections.Counter()
    with open(manifest_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            cnt[r['label']] += 1
    # order by label index (0..C-1)
    ordered = sorted(label_map, key=lambda k: label_map[k])
    weights = []
    for lab in ordered:
        w = 1.0 / max(1, cnt[lab])
        weights.append(w)
    s = sum(weights)
    weights = [w * len(weights) / s for w in weights]  # normalize
    return torch.tensor(weights, dtype=torch.float32)


class SpeechDataset(Dataset):
    """
    Reads manifest CSV with columns: path,label,speaker,split
    Example row:
      data/wavs/M_10_EVAN_S_9_NEUTRAL_5.wav,neutral,evan,train
    """

    def __init__(self, manifest, split, label_map, sr=16000, min_dur=0.5, max_dur=20.0):
        self.sr = int(sr)
        self.label_map = label_map
        self.items = []

        with open(manifest, newline="", encoding="utf-8") as f:
            # try to use header if present; otherwise fall back
            reader = csv.DictReader(f)
            has_header = reader.fieldnames is not None and {"path", "label", "split"}.issubset(set(reader.fieldnames))
            if not has_header:
                f.seek(0)
                reader = csv.DictReader(f, fieldnames=["path", "label", "speaker", "split"])

            for r in reader:
                # skip header row if we forced fieldnames (avoid treating header as data)
                if not has_header and r["path"] == "path":
                    continue

                if r.get("split", "").strip().lower() != split:
                    continue
                path = r.get("path")
                lab_name = r.get("label")
                if not path or not lab_name or lab_name not in self.label_map:
                    continue

                # quick duration check using metadata (cheap)
                try:
                    info = sf.info(path)
                    dur_meta = float(info.frames) / float(info.samplerate)
                except Exception:
                    continue

                if dur_meta < min_dur or dur_meta > max_dur:
                    continue

                self.items.append((path, self.label_map[lab_name]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, lab = self.items[i]
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        wav = _safe_mono_float32(wav)

        # resample if needed
        if sr != self.sr:
            try:
                wav = librosa.resample(y=wav, orig_sr=sr, target_sr=self.sr)
            except TypeError:
                wav = librosa.resample(wav, sr, self.sr)
            wav = _safe_mono_float32(wav)

        # ---------- augmentation ----------
        if getattr(self, "augment", False):
            wav = self._augment(wav)

        # sanitize after augmentation
        wav = np.clip(wav, -1.0, 1.0).astype(np.float32, copy=False)
        dur = float(len(wav)) / float(self.sr)  # duration AFTER augment (important!)

        return {
            "wav": torch.from_numpy(wav),
            "label": torch.tensor(lab, dtype=torch.long),
            "dur": dur,
            "path": path,
        }

    def _augment(self, wav: np.ndarray) -> np.ndarray:
        # small, prosody-safe augmentations
        if np.random.rand() < 0.3:  # ±1.5 dB gain
            wav = wav * (10 ** (np.random.uniform(-1.5, 1.5) / 20))
        if np.random.rand() < 0.3:  # tiny white noise
            wav = wav + np.random.randn(len(wav)).astype('float32') * 0.005
        if np.random.rand() < 0.2:  # ±3% speed (changes duration → we recompute dur later)
            rate = float(np.random.uniform(0.97, 1.03))
            # librosa can throw for very short arrays; guard it
            if len(wav) > int(0.05 * self.sr):
                wav = librosa.effects.time_stretch(wav, rate=rate)
        return wav


class Collator:
    """
    Collates variable-length waveforms for Wav2Vec2Processor.
    Ensures each item is 1-D mono float32 numpy array.
    """

    def __init__(self, backbone_id: str, sample_rate: int = 16000):
        self.processor = AutoProcessor.from_pretrained(backbone_id)
        self.sample_rate = int(sample_rate)

    def __call__(self, batch):
        wavs = [b["wav"] for b in batch]
        labels = [int(b["label"]) for b in batch]
        durs = [float(b["dur"]) for b in batch]

        # sanitize every wav to mono, 1-D, float32
        wavs_clean = []
        for w in wavs:
            # if tensor, convert to numpy for processor
            if isinstance(w, torch.Tensor):
                w = w.detach().cpu().numpy()
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

        input_values = X["input_values"]                      # (B, T) float32
        attention_mask = X.get("attention_mask", torch.ones_like(input_values, dtype=torch.long))

        return {
            "input_values": input_values,                     # (B, T)
            "attention_mask": attention_mask,                 # (B, T)
            "labels": torch.tensor(labels, dtype=torch.long), # (B,)
            "lengths_sec": durs,                              # list of floats (post-augment durations)
        }
