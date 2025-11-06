# Build a starter manifest from filenames like F_01_OISHI_S_1_ANGRY_1.wav
import os, csv
ROOT='data/wavs'; OUT='data/manifest.csv'
MAP={'ANGRY':'anger','DISGUST':'disgust','FEAR':'fear','HAPPY':'happy','NEUTRAL':'neutral','SAD':'sad','SURPRISE':'surprise'}
rows=[]
for fn in os.listdir(ROOT):
    if not fn.lower().endswith('.wav'): continue
    emo=None
    for k,v in MAP.items():
        if k in fn.upper(): emo=v; break
    if not emo: continue
    parts=fn.split('_'); speaker=parts[2].lower() if len(parts)>=3 else 'spk'
    rows.append([f'{ROOT}/{fn}', emo, speaker, 'train'])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT,'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['path','label','speaker','split']); w.writerows(rows)
print(f'Wrote {len(rows)} rows to {OUT}. Now edit split column for train/val/test.')
