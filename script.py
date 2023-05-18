import subprocess
import os 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
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

def get_length_distr(fname):
    df = pd.read_csv(fname)
    df_begin = df["start"].to_list()
    df_end = df["end"].to_list()
    lengths = [ end - beg for beg, end in zip(df_begin, df_end) ]
    plt.hist(lengths)
    plt.ylim((-1, 30000))
    plt.show()

def get_expected_n_variance(csv, col_name, lang_filter=None, lang_col='lang'):
    df = pd.read_csv(csv)
    if lang_filter:
        df = df[df[lang_col] == lang_filter]
    data = df[col_name].to_numpy()

    return data.mean(), data.std(), len(data)

def replace_empty(csv, col_name, replacement):
    df = pd.read_csv(csv)
    df.to_csv(csv.replace('.csv', '_old.csv'))
    df[col_name] = df[col_name].replace(np.nan, replacement)
    df.to_csv(csv)

dirs = [
    'refs/asr/audio_to_release/ckt',
    'refs/asr/audio_to_release/evn',
    'refs/asr/audio_to_release/mhr',
    'refs/asr/audio_to_release/sah',
    'refs/asr/audio_to_release/yrk'
]

replace_empty(
    'models/wav2vec2-large-xlsr-japlmthufielta-ipa-plus-2000/evaluated.csv',
    'phonetic_ev',
    1
)

""" pprint(get_expected_n_variance(
    'models/wav2vec2-xlsr-multilingual-56/out.csv',
    'mos_pred',
    lang_filter='ckt'
)) """