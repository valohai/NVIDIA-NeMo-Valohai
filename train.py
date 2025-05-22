import argparse
import json
import os

import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
import valohai


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


train_manifest = valohai.inputs("train_manifest").path()
val_manifest = valohai.inputs("val_manifest").path()

print(f"Train manifest: {train_manifest}")
print(f"Validation manifest: {val_manifest}")


def flatten_audio_paths(file_path, new_input_dir_name):
    updated_lines = []
    with open(file_path, "r") as infile:
        for line in infile:
            data = json.loads(line)
            # Extract the filename only (e.g., 1988-24833-0000.wav)
            filename = os.path.basename(data["audio_filepath"])
            # Build the new path under the correct Valohai input mount
            data["audio_filepath"] = f"/valohai/inputs/{new_input_dir_name}/{filename}"
            updated_lines.append(json.dumps(data))

    with open(file_path, "w") as outfile:
        outfile.write("\n".join(updated_lines) + "\n")


flatten_audio_paths(train_manifest, "train_input")
flatten_audio_paths(val_manifest, "val_input")

args = parse_args()

batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate

# Load pretrained QuartzNet15x5 model
model = nemo_asr.models.EncDecCTCModel.from_pretrained("stt_en_quartznet15x5")

# Set up training data
model.setup_training_data(
    train_data_config={
        "manifest_filepath": train_manifest,
        "sample_rate": 16000,
        "batch_size": batch_size,
        "shuffle": True,
        "labels": model.decoder.vocabulary,
    },
)

# Set up validation data
model.setup_validation_data(
    val_data_config={
        "manifest_filepath": val_manifest,
        "sample_rate": 16000,
        "batch_size": batch_size,
        "shuffle": False,
        "labels": model.decoder.vocabulary,
    },
)

# Set up optimizer and learning rate scheduler
model.setup_optimization(
    optim_config={
        "lr": learning_rate,
        "sched": {
            "name": "CosineAnnealing",
            "warmup_steps": 500,
        },
    },
)

# Initialize the PyTorch Lightning trainer
trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=epochs,
    precision="16-mixed",
)

# Run training
trainer.fit(model)

print("Model training completed")

# Save the trained model
print("Saving the trained model...")

output_dir_path = os.getenv("VH_OUTPUTS_DIR", ".outputs")
# Save the trained model
output_path = os.path.join(output_dir_path, "QuartzNet15x5-LibriSpeech-finetuned.nemo")
model.save_to(output_path)

print("Saved completed artifacts to outputs")
