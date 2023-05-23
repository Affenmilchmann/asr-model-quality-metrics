import subprocess
import os 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
from pprint import pprint

from DataQuality import Evaluator

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

def get_sigma_ratio(csv, col_name, sigma_count = 1):
    avg, sigma, _ = get_expected_n_variance(csv, col_name)
    df = pd.read_csv(csv)
    data = df[col_name].to_numpy()
    left = avg - sigma * sigma_count
    right = avg + sigma * sigma_count
    in_count = sum([ 1 if left < x < right else 0 for x in data])
    return in_count / len(data), (left, right)

def replace_empty(csv, col_name, replacement):
    df = pd.read_csv(csv)
    df.to_csv(csv.replace('.csv', '_old.csv'))
    df[col_name] = df[col_name].replace(np.nan, replacement)
    df.to_csv(csv)

def combined_mos_phoned_scatter(csv_file, lang_col="lang", mos_col="mos_pred", eval_col="phonetic_ev", img_dir="", title=None):
    color_wheel = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    data_by_lang = Evaluator.load_by_lang(
        csv_file=csv_file,
        lang_col=lang_col,
        mos_col=mos_col,
        eval_col=eval_col
    )
    del data_by_lang['all_langs']
    stat_by_lang = Evaluator.statistic_by_lang(
        csv_file=csv_file,
        lang_col=lang_col,
        mos_col=mos_col,
        eval_col=eval_col
    )
    fig, axes = plt.subplots(2, 2, dpi=200, figsize=(10, 8))
    axes[0, 0].set_title('total')
    axes[0, 0].set_xlim((0., 1.05))
    axes[0, 0].set_ylim((0., 5.))
    for i, lang in enumerate(data_by_lang.keys()):
        axes[0, 0].scatter(data_by_lang[lang]["eval"], data_by_lang[lang]["mos"], s=1, c=color_wheel[i])
    text = 'n: {}\ncorrelation: {}\np-value: {:e}'.format(
        round(stat_by_lang['all_langs'][2], 3),
        round(stat_by_lang['all_langs'][0], 3),
        stat_by_lang['all_langs'][1]
    )
    axes[0, 0].text(0.05, 0.3, text, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    del stat_by_lang['all_langs']

    for i, lang in enumerate(data_by_lang.keys()):
        print(((i + 1) // 2, (i + 1) % 2))
        axes[(i + 1) // 2, (i + 1) % 2].scatter(data_by_lang[lang]["eval"], data_by_lang[lang]["mos"], s=1, c=color_wheel[i])
        axes[(i + 1) // 2, (i + 1) % 2].set_title(lang)
        text = 'n: {}\ncorrelation: {}\np-value: {:e}'.format(
            round(stat_by_lang[lang][2], 3),
            round(stat_by_lang[lang][0], 3),
            stat_by_lang[lang][1]
        )
        axes[(i + 1) // 2, (i + 1) % 2].text(0.05, 0.3, text, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        axes[(i + 1) // 2, (i + 1) % 2].set_xlim((0., 1.05))
        axes[(i + 1) // 2, (i + 1) % 2].set_ylim((0., 5.))
    plt.tight_layout()
    fig.suptitle(title, fontweight='bold', fontsize=13)
    fig.supxlabel('Phonetic Edit Distance')
    fig.supylabel('Predicted MOS')
    plt.subplots_adjust(bottom=0.07, top=0.93, left=0.06)
    plt.savefig(f"{img_dir}/combined.png")

dirs = [
    'refs/asr/audio_to_release/ckt',
    'refs/asr/audio_to_release/evn',
    'refs/asr/audio_to_release/mhr',
    'refs/asr/audio_to_release/sah',
    'refs/asr/audio_to_release/yrk'
]

pprint(get_sigma_ratio(
    'models/wav2vec2-base-960h/out.csv',
    'mos_pred',
    sigma_count=2
))

""" pprint(get_expected_n_variance(
    'models/wav2vec2-base-960h/out.csv',
    'mos_pred'
)) """

""" combined_mos_phoned_scatter(
    'models/wav2vec2-base/out.csv',
    img_dir='models/wav2vec2-base/img',
    title='wav2vec2-base'
) """