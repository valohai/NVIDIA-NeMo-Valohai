# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# USAGE: python get_librispeech_data.py --data_root=<where to put data>
#        --data_set=<datasets_to_download> --num_workers=<number of parallel workers>
# where <datasets_to_download> can be: dev_clean, dev_other, test_clean,
# test_other, train_clean_100, train_clean_360, train_other_500 or ALL
# You can also put more than one data_set comma-separated:
# --data_set=dev_clean,train_clean_100
import argparse
import fnmatch
import functools
import json
import logging
import multiprocessing
import os
import pathlib
import subprocess
import tarfile
import tempfile
import urllib.request

from tqdm import tqdm

parser = argparse.ArgumentParser(description="LibriSpeech Data download")
parser.add_argument("--data_root", required=True, default=None, type=str)
parser.add_argument("--temp_root", default="./temp", type=str)
parser.add_argument("--data_sets", default="dev_clean", type=str)
parser.add_argument("--num_workers", default=max(4, multiprocessing.cpu_count() + 1), type=int)
args = parser.parse_args()

URLS = {
    "TRAIN_CLEAN_100": ("http://www.openslr.org/resources/12/train-clean-100.tar.gz"),
    "TRAIN_CLEAN_360": ("http://www.openslr.org/resources/12/train-clean-360.tar.gz"),
    "TRAIN_OTHER_500": ("http://www.openslr.org/resources/12/train-other-500.tar.gz"),
    "DEV_CLEAN": "http://www.openslr.org/resources/12/dev-clean.tar.gz",
    "DEV_OTHER": "http://www.openslr.org/resources/12/dev-other.tar.gz",
    "TEST_CLEAN": "http://www.openslr.org/resources/12/test-clean.tar.gz",
    "TEST_OTHER": "http://www.openslr.org/resources/12/test-other.tar.gz",
    "DEV_CLEAN_2": "https://www.openslr.org/resources/31/dev-clean-2.tar.gz",
    "TRAIN_CLEAN_5": "https://www.openslr.org/resources/31/train-clean-5.tar.gz",
}


def __retrieve_with_progress(source: str, filename: str):
    """
    Downloads source to destination
    Displays progress bar
    Args:
        source: url of resource
        destination: local filepath
    Returns:
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as f:
        response = urllib.request.urlopen(source)
        total = response.length

        if total is None:
            f.write(response.content)
        else:
            with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                for data in response:
                    f.write(data)
                    pbar.update(len(data))


def __maybe_download_file(destination: str, url: str):
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        url: url of resource
    Returns:
    """
    if not os.path.exists(destination):
        logging.info("%s does not exist, downloading from %s", destination, url)
        __retrieve_with_progress(url, filename=destination + ".tmp")

        os.rename(destination + ".tmp", destination)
        logging.info("Downloaded %s", destination)
    else:
        logging.info("Destination %s exists. Skipping.", destination)
    return destination


def __process_transcript(file_path: str, dst_folder: str, rel_root: str):
    entries = []
    root = os.path.dirname(file_path)
    with open(file_path, encoding="utf-8") as fin:
        for line in fin:
            id, text = line[: line.index(" ")], line[line.index(" ") + 1 :]
            transcript_text = text.lower().strip()

            flac_file = os.path.join(root, id + ".flac")
            wav_file = os.path.join(dst_folder, id + ".wav")
            # Convert to 1-channel FLAC
            subprocess.check_call(["sox", flac_file, "-c", "1", wav_file])
            # Grab duration
            duration_output = subprocess.check_output(["sox", "--info", "-D", wav_file], text=True)
            duration = float(duration_output.strip())
            entry = {
                "audio_filepath": os.path.relpath(wav_file, rel_root),
                "duration": duration,
                "text": transcript_text,
            }
            entries.append(entry)

    return entries


def __process_data(source_folder: str, dst_folder: str, manifest_file: str, num_workers: int):
    """
    Converts flac to wav and build manifests's json
    Args:
        source_folder: source with flac files and transcripts
        dst_folder: where wav files will be stored
        manifest_file: where to store manifest
        num_workers: number of parallel workers processing files
    Returns:
    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    files = []

    for root, _dirnames, filenames in os.walk(source_folder):
        for filename in fnmatch.filter(filenames, "*.trans.txt"):
            files.append(os.path.join(root, filename))

    rel_root = os.path.dirname(manifest_file)

    with multiprocessing.Pool(num_workers) as p, open(manifest_file, "w") as fout:
        processing_func = functools.partial(__process_transcript, dst_folder=dst_folder, rel_root=rel_root)
        results = p.imap(processing_func, files)
        for result in tqdm(results, total=len(files)):
            for m in result:
                print(json.dumps(m), file=fout)


def main():
    data_sets = args.data_sets
    num_workers = args.num_workers
    data_root = pathlib.Path(args.data_root)
    temp_root = pathlib.Path(args.temp_root)
    download_root = temp_root / "download"
    download_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    if data_sets == "ALL":
        data_sets = "dev_clean,dev_other,train_clean_100,train_clean_360,train_other_500,test_clean,test_other"
    if data_sets == "mini":
        data_sets = "dev_clean_2,train_clean_5,test_clean"
    for data_set in data_sets.split(","):
        logging.info("Working on: %s", data_set)
        tarball_path = download_root / (data_set + ".tar.gz")
        __maybe_download_file(str(tarball_path), url=URLS[data_set.upper()])
        with tempfile.TemporaryDirectory(dir=temp_root) as extract_path:
            with tarfile.open(tarball_path) as tf:
                tf.extractall(extract_path)
            __process_data(
                source_folder=str(extract_path),
                dst_folder=str(data_root / "LibriSpeech" / data_set.replace("_", "-")),
                manifest_file=str(data_root / (data_set + ".json")),
                num_workers=num_workers,
            )
    logging.info("Done!")


if __name__ == "__main__":
    main()
