import argparse
import json
import os

import nemo.collections.asr as nemo_asr
import torch.mps
import valohai
from nemo.collections.asr.metrics.wer import word_error_rate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=valohai.inputs("model").path())
    parser.add_argument("--test_manifest_path", default=valohai.inputs("test_manifest").path())
    parser.add_argument("--output_dir", default=os.getenv("VH_OUTPUTS_DIR", ".outputs"))
    parser.add_argument("--audio_base_dir", default="/valohai/inputs/test_input")
    return parser.parse_args()


def load_manifest(manifest_path: str, audio_base_dir: str):
    print(f"Using test manifest: {manifest_path}")
    audio_filepaths = []
    ground_truths = []
    with open(manifest_path, "r") as f:
        for line in f:
            item = json.loads(line)
            file_path = os.path.join(audio_base_dir, os.path.basename(item["audio_filepath"]))
            audio_filepaths.append(file_path)
            ground_truths.append(item["text"])
    return audio_filepaths, ground_truths


def main():
    args = parse_args()
    model_path = args.model_path
    test_manifest_path = args.test_manifest_path
    output_dir = args.output_dir
    audio_base_dir = args.audio_base_dir
    if not (model_path and os.path.exists(model_path)):
        raise ValueError(f"Model path does not exist: {model_path}")
    if not (test_manifest_path and os.path.exists(test_manifest_path)):
        raise ValueError(f"Test manifest path does not exist: {test_manifest_path}")
    if not os.path.exists(audio_base_dir):
        raise ValueError(f"Audio base directory does not exist: {audio_base_dir}")

    audio_filepaths, ground_truths = load_manifest(test_manifest_path, audio_base_dir=audio_base_dir)

    print(f"Loading model from: {model_path}")
    asr_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
    asr_model.eval()
    try:
        if torch.mps.is_available():
            asr_model.to("mps")
    except Exception:  # might not even have the mps module
        asr_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Running inference on {len(audio_filepaths)} files...")
    predictions = asr_model.transcribe(paths2audio_files=audio_filepaths, batch_size=16)

    wer_score = word_error_rate(predictions, ground_truths)
    print(f"Word Error Rate (WER): {wer_score:.4f}")

    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "wer": wer_score,
                "predictions": predictions,
                "ground_truths": ground_truths,
            },
            f,
            indent=2,
        )

    print(f"Saved evaluation results to {results_path}")


if __name__ == "__main__":
    main()
