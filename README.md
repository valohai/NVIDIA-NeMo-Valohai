# NeMo ASR Training Pipeline

This repository contains a complete pipeline for fine-tuning NVIDIA's NeMo QuartzNet model for Automatic Speech Recognition (ASR) using the LibriSpeech dataset, orchestrated with Valohai.

## Overview

This project demonstrates how to:
1. Download and preprocess the LibriSpeech dataset
2. Fine-tune a pre-trained QuartzNet15x5 model
3. Evaluate the model's performance
4. Orchestrate the entire workflow using Valohai


## Project Structure

- `prepare-dataset.py`: Downloads and preprocesses LibriSpeech data
- `train.py`: Fine-tunes the QuartzNet model
- `evaluate.py`: Evaluates model performance using Word Error Rate (WER)
- `valohai.yaml`: Defines the Valohai pipeline and execution steps
- `requirements.txt`: Core Python dependencies


## Pipeline Steps

1. **Prepare Dataset**
   - Downloads a subset of LibriSpeech ("mini" version)
   - Converts FLAC files to WAV
   - Creates manifest files for training, validation, and testing

2. **Train Model**
   - Fine-tunes the pre-trained QuartzNet15x5 model
   - Uses the train and validation manifests
   - Configurable epochs, learning rate, and batch size

3. **Evaluate Model**
   - Calculates Word Error Rate (WER) on the test set
   - Generates predictions and compares with ground truth

## Dataset

This project uses LibriSpeech, a corpus of approximately 1000 hours of 16kHz read English speech. The pipeline is configured to use the "mini" subset by default, which includes:
- `dev-clean-2`: A small development set
- `train-clean-5`: A small training set (5 hours)
We customized the dataset to include a test set for the evaluation step and used:
- `test-clean`: The standard test set

## Model

The pipeline fine-tunes NVIDIA's QuartzNet15x5, a convolutional neural network for speech recognition that achieves state-of-the-art results on LibriSpeech.

## License

This project uses code from NVIDIA's NeMo toolkit, which is licensed under the Apache License 2.0.

## Acknowledgments

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [LibriSpeech Dataset](http://www.openslr.org/12/)
- [Valohai](https://valohai.com/)
