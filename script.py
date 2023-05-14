import subprocess
import os 
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pprint import pprint

def get_wav_len(fname):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", fname
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout)

def get_dir_total(dir_path):
    dir_path = Path(dir_path)
    files = os.listdir(dir_path)
    total = 0
    for f in tqdm(files):
        total += get_wav_len(dir_path.joinpath(f))
    return total

def get_csv_total_by_lang(csv_path):
    df = pd.read_csv(csv_path)
    langs = df["lang"].unique()
    result = {}
    for l in langs:
        l_df = df[df["lang"] == l]
        total = 0
        for _, row in l_df.iterrows():
            total += row['end'] - row['start']
        result[l] = total
    return result

dirs = [
    'refs/asr/audio_to_release/ckt',
    'refs/asr/audio_to_release/evn',
    'refs/asr/audio_to_release/mhr',
    'refs/asr/audio_to_release/sah',
    'refs/asr/audio_to_release/yrk'
]
pprint(get_csv_total_by_lang('refs/asr/asr_data.csv'))