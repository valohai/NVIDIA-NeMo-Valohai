import argparse
import json
import os
import tempfile

import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
import valohai


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_manifest", default=valohai.inputs("train_manifest").path())
    parser.add_argument("--val_manifest", default=valohai.inputs("val_manifest").path())
    parser.add_argument("--train_input", default="/valohai/inputs/train_input")
    parser.add_argument("--val_input", default="/valohai/inputs/val_input")
    parser.add_argument("--output_dir_path", default=os.getenv("VH_OUTPUTS_DIR", ".outputs"))
    return parser.parse_args()


def flatten_audio_paths(input_file_path: str, output_file_path: str, new_input_dir_name: str):
    with open(input_file_path, "r") as infile, open(output_file_path, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            # Extract the filename only (e.g., 1988-24833-0000.wav)
            filename = os.path.basename(data["audio_filepath"])
            # Build the new path under the correct Valohai input mount
            data["audio_filepath"] = os.path.join(new_input_dir_name, filename)
            print(json.dumps(data), file=outfile)


def train(
    *,
    train_manifest: str,
    val_manifest: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    output_dir_path: str,
) -> None:
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
    # Save the trained model
    output_path = os.path.join(output_dir_path, "QuartzNet15x5-LibriSpeech-finetuned.nemo")
    model.save_to(output_path)
    print("Saved completed artifacts to outputs")


def main() -> None:
    args = parse_args()
    if not (args.train_manifest and os.path.exists(args.train_manifest)):
        raise ValueError(f"Train manifest path does not exist: {args.train_manifest}")
    if not (args.val_manifest and os.path.exists(args.val_manifest)):
        raise ValueError(f"Validation manifest path does not exist: {args.val_manifest}")
    if not os.path.exists(args.train_input):
        raise ValueError(f"Train input path does not exist: {args.train_input}")
    if not os.path.exists(args.val_input):
        raise ValueError(f"Validation input path does not exist: {args.val_input}")

    with (
        tempfile.NamedTemporaryFile(suffix=".train.json", delete=False) as train_manifest_tf,
        tempfile.NamedTemporaryFile(suffix=".val.json", delete=False) as val_manifest_tf,
    ):
        train_manifest = train_manifest_tf.name
        val_manifest = val_manifest_tf.name
        flatten_audio_paths(args.train_manifest, train_manifest, args.train_input)
        flatten_audio_paths(args.val_manifest, val_manifest, args.val_input)

    train(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir_path=args.output_dir_path,
    )


if __name__ == "__main__":
    main()
