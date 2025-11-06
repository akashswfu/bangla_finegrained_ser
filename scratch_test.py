from transformers import Wav2Vec2Model
print("import OK")
m = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", use_safetensors=True, local_files_only=False)
print("model loaded:", m.__class__.__name__)
