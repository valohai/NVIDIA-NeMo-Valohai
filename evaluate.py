import os
import json
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
import valohai

# === Valohai inputs ===
model_path = valohai.inputs('model').path()
test_manifest_path = valohai.inputs('test_manifest').path()

print(f"Loading model from: {model_path}")
print(f"Using test manifest: {test_manifest_path}")

# === Load trained model ===
asr_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
asr_model.eval()
asr_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# === Read test samples ===
audio_filepaths = []
ground_truths = []

with open(test_manifest_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        audio_filepaths.append(item['audio_filepath'])
        ground_truths.append(item['text'])

# === Inference ===
print(f"Running inference on {len(audio_filepaths)} files...")
predictions = asr_model.transcribe(paths2audio_files=audio_filepaths, batch_size=16)

# === Compute WER ===
wer_score = word_error_rate(predictions, ground_truths)
print(f"Word Error Rate (WER): {wer_score:.4f}")

# === Save outputs ===
output_dir = os.getenv('VH_OUTPUTS_DIR', '.outputs')
os.makedirs(output_dir, exist_ok=True)

# Save predictions + references
results_path = os.path.join(output_dir, 'evaluation_results.json')
with open(results_path, 'w') as f:
    json.dump({
        "wer": wer_score,
        "predictions": predictions,
        "ground_truths": ground_truths,
    }, f, indent=2)

print(f"Saved evaluation results to {results_path}")
