import pandas as pd
import numpy as np
import os
import scipy
from pathlib import Path
from math import ceil
from matplotlib import pyplot as plt
from abydos import distance
from pprint import pprint
from pathlib import Path

from refs.nisqa.nisqa.NISQA_model import nisqaModel

class Evaluator():
    mos_only_model = "refs/nisqa/weights/nisqa_mos_only.tar"
    nisqa_default_args = {
        'ms_channel': None,
        'output_dir': None,
        'ms_max_segments': 1500,
    }
    phonetic = distance.PhoneticEditDistance()

    @classmethod
    def eval_quality_single_wav(cls, audio_path):
        """Predict MOS of the audio usin NISQA model

        Args:
            audio_path (str): path to wav file

        Returns:
            int: predicted MOS score
        """
        nisqa_args = {
            'mode': 'predict_file',
            'pretrained_model': cls.mos_only_model,
            'deg': audio_path
        }
        nisqa_args.update(cls.nisqa_default_args)

        nisqa = nisqaModel(nisqa_args)

        df = nisqa.predict()

        return df['mos_pred'][0]
    
    @classmethod
    def eval_quality_from_csv(cls, csv_file, path_col, save_to=None):
        """Predict MOS of the audio using NISQA model. Takes audio paths from the given csv file by the given column.

        Args:
            csv_file (str): path to the csv file
            path_col (str): name of the csv column with paths to audios
            save_to (str): path where to save csv. Will save nothing if None. Default is None

        Returns:
            DataFrame: dataframe with new predicted mos_pred column added
        """
        if '/' in csv_file or '\\' in csv_file:
            path = Path(csv_file)
            parent = path.parent
            csv_file = path.name 
        else:
            parent = None
        
        nisqa_args = {
            'mode': 'predict_csv',
            'pretrained_model': cls.mos_only_model,
            'csv_deg': path_col,
            'csv_file': csv_file,
            'data_dir': parent if parent else ''
        }
        nisqa_args.update(cls.nisqa_default_args)

        nisqa = nisqaModel(nisqa_args)

        df = nisqa.predict()

        df.reset_index()
        if save_to:
            print(f"saving to {save_to}")
            df.to_csv(save_to, index=False)

        return df['mos_pred']
    
    @classmethod
    def eval_asr(cls, predicted_text, original_text):
        return cls.phonetic.dist(predicted_text, original_text)

    @classmethod
    def load_by_lang(cls, csv_file, lang_col="lang", mos_col="mos_pred", eval_col="phonetic_ev"):
        df = pd.read_csv(csv_file)
        langs = np.append(df[lang_col].unique(), 'all_langs')
        by_lang = {}
        for lang in langs:
            if lang == 'all_langs':
                lang_df = df
            else:
                lang_df = df[df[lang_col] == lang]
            by_lang[lang] = {
                "mos": lang_df[mos_col].to_numpy(),
                "eval": lang_df[eval_col].to_numpy()
            }
        return by_lang

    @classmethod
    def statistic_by_lang(cls, csv_file, lang_col="lang", mos_col="mos_pred", eval_col="phonetic_ev"):
        data_by_lang = cls.load_by_lang(
            csv_file=csv_file,
            lang_col=lang_col,
            mos_col=mos_col,
            eval_col=eval_col
        )
        stat_by_lang = {}
        for lang in data_by_lang.keys():
            stat_by_lang[lang] = scipy.stats.pearsonr(data_by_lang[lang]["mos"], data_by_lang[lang]["eval"])
            stat_by_lang[lang] = (stat_by_lang[lang][0], stat_by_lang[lang][1], len(data_by_lang[lang]["mos"]))
        return stat_by_lang

    @classmethod
    def hist_by_model(cls, model_name):
        models_path = Path('models/')
        model_dir = models_path.joinpath(model_name)
        evaluated_csv = model_dir.joinpath('evaluated.csv')
        
        df = pd.read_csv(evaluated_csv)
        phon_ev = df['phonetic_ev'].to_numpy()
        plt.hist(phon_ev, bins=50)
        plt.title(f"{model_name} model")
        plt.xlabel("PhoneticEditDistance")
        plt.ylabel("n")
        plt.savefig(model_dir.joinpath('img/hist.png'))
        plt.close()

    @classmethod
    def plot_by_lang(cls, csv_file, lang_col="lang", mos_col="mos_pred", eval_col="phonetic_ev", img_dir="", title=None):
        color_wheel = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        data_by_lang = cls.load_by_lang(
            csv_file=csv_file,
            lang_col=lang_col,
            mos_col=mos_col,
            eval_col=eval_col
        )
        del data_by_lang['all_langs']
        stat_by_lang = cls.statistic_by_lang(
            csv_file=csv_file,
            lang_col=lang_col,
            mos_col=mos_col,
            eval_col=eval_col
        )
        ax_all = plt.subplot()
        ax_all.set_title('all_langs' + f' ({title})' if title else '')
        ax_all.set_xlabel("Evaluated PhoneticEditDistance")
        ax_all.set_ylabel("Predicted MOS")
        ax_all.set_xlim((0., 1.1))
        ax_all.set_ylim((0., 5.))
        for i, lang in enumerate(data_by_lang.keys()):
            ax_all.scatter(data_by_lang[lang]["eval"], data_by_lang[lang]["mos"], s=1, c=color_wheel[i])
        text = 'n: {}\ncorrelation: {}\np-value: {:e}'.format(
            round(stat_by_lang['all_langs'][2], 3),
            round(stat_by_lang['all_langs'][0], 3),
            stat_by_lang['all_langs'][1]
        )
        ax_all.text(0.05, 0.3, text, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        ax_all.figure.savefig(Path(img_dir).joinpath('all_langs.png'))
        ax_all.clear()

        del stat_by_lang['all_langs']

        for i, lang in enumerate(data_by_lang.keys()):
            ax = plt.subplot()
            ax.scatter(data_by_lang[lang]["eval"], data_by_lang[lang]["mos"], s=1, c=color_wheel[i])
            ax.set_title(lang + f' ({title})' if title else '')
            ax.set_xlabel("Evaluated PhoneticEditDistance")
            ax.set_ylabel("Predicted MOS")
            text = 'n: {}\ncorrelation: {}\np-value: {:e}'.format(
                round(stat_by_lang[lang][2], 3),
                round(stat_by_lang[lang][0], 3),
                stat_by_lang[lang][1]
            )
            ax.text(0.05, 0.3, text, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
            ax.set_xlim((0., 1.1))
            ax.set_ylim((0., 5.))
            ax.figure.savefig(Path(img_dir).joinpath(f'{lang}.png'))
            ax.clear()     

    @classmethod
    def analyse_track_asr(cls, track_path, predicted_text, original_text):
        quality_metric = cls.eval_quality_single_wav(track_path)
        asr_metric = cls.eval_asr(predicted_text, original_text)
        return quality_metric, asr_metric
    
    #@classmethod
    #def present_analysis(cls, track_paths, texts, predicted_texts, track_analysis???)

evaluated_file = Path('refs/asr/evaluated.csv')
model_name = 'wav2vec2-large-TIMIT-IPA'

models_dir = Path('models/')
model_dir = models_dir.joinpath(model_name)
if not (model_dir.is_dir()): model_dir.mkdir()
img_dir = model_dir.joinpath('img')
if not (img_dir.is_dir()): img_dir.mkdir()

if not model_dir.joinpath('evaluated.csv').is_file():
    os.rename(evaluated_file, model_dir.joinpath('evaluated.csv'))
evaluated_file = model_dir.joinpath('evaluated.csv')

Evaluator.hist_by_model(model_name)

Evaluator.eval_quality_from_csv(
    csv_file=str(evaluated_file),
    path_col='new_path',
    save_to=model_dir.joinpath('out.csv')
)
pprint(
    Evaluator.plot_by_lang(
        model_dir.joinpath('out.csv'),
        img_dir=img_dir,
        title=f'speech31/{model_name} model'
    )
)

#pprint(evltr.eval_quality("refs/asr/new_audio/0.wav"))
