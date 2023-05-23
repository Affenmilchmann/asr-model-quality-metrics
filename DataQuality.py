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

        df.reset_index(drop=True)
        if save_to:
            print(f"saving to {save_to}")
            df.to_csv(save_to, index=False)

        return df['mos_pred']
    
    @classmethod
    def eval_asr(cls, predicted_text, original_text):
        """Evaluate Phonetic Edit Distance metric for two texts.

        Args:
            predicted_text (str): text predicted by a model
            original_text (str): reference text

        Returns:
            float: [0, 1]
        """
        return cls.phonetic.dist(predicted_text, original_text)

    @classmethod
    def __load_by_lang(cls, csv_file, lang_col="lang", mos_col=None, eval_col=None):
        """Loads data from csv and groups it by lang.

        Args:
            csv_file (str): path to csv file.

            lang_col (str, optional): Name of the csv column that contains lang. Defaults to "lang".

            mos_col (str, optional): Name of the csv column that contains MOS data. If None, MOS will not be returned. Defaults to None.

            eval_col (str, optional): Name of the csv column that contains PhonED data. If None, PhonED will not be returned. Defaults to None.

        Returns:
            dict[str][str] : ndarray

            First key stands for language, second stands for "mos" or "eval".

            Example: {
                "ckt": {
                    "mos": ndarray[1, 2, ..., 3],
                    "eval": ndarray[0.1, 0.2, ..., 0.3]
                }
            }
        """
        df = pd.read_csv(csv_file)
        langs = np.append(df[lang_col].unique(), 'all_langs')
        by_lang = {}
        for lang in langs:
            if lang == 'all_langs':
                lang_df = df
            else:
                lang_df = df[df[lang_col] == lang]
            by_lang[lang] = {
                "mos": lang_df[mos_col].to_numpy() if mos_col else None,
                "eval": lang_df[eval_col].to_numpy() if eval_col else None
            }
        return by_lang

    @classmethod
    def statistic_by_lang(cls, csv_file, lang_col="lang", mos_col="mos_pred", eval_col="phonetic_ev"):
        """Get correlation and p-value of MOS and PhonED distribution

        Args:
            csv_file (str): path to csv file.

            lang_col (str, optional): Name of the csv column that contains lang. Defaults to "lang".

            mos_col (str, optional): Name of the csv column that contains MOS data. If None, MOS will not be returned. Defaults to None.

            eval_col (str, optional): Name of the csv column that contains PhonED data. If None, PhonED will not be returned. Defaults to None.

        Returns:
            tuple[float, float, int]: correlation, p-value, number of samples
        """
        data_by_lang = cls.__load_by_lang(
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
    def hist_phoned_by_model(cls, model_name):
        """Plot a histogram of PhonED. Takes evaluated.csv from model's folder.
        It must be precomputed with eval_quality_from_csv and eval_phoned_from_csv.
        Saves histogram as png to /models/<model_name>/img/hist.png

        Args:
            model_name (str): name of the model. Name of the model's folder in /models/ dir
        """
        models_path = Path('models/')
        model_dir = models_path.joinpath(model_name)
        evaluated_csv = model_dir.joinpath('evaluated.csv')
        
        df = pd.read_csv(evaluated_csv)
        phon_ev = df['phonetic_ev'].to_numpy()
        plt.hist(phon_ev, bins=50)
        plt.title(f"{model_name} model")
        plt.xlabel("PhoneticEditDistance")
        plt.ylabel("n")
        plt.tight_layout()
        plt.savefig(model_dir.joinpath('img/hist.png'))
        plt.close()

    @classmethod
    def plot_by_lang(cls, csv_file, lang_col="lang", mos_col="mos_pred", eval_col="phonetic_ev", img_dir="", title=None):
        """Plot a scatter plot MOS x PhonED for each lang. Saves plot to <img_dir>/<lang>.png 
        Also creates an additional scatter with all languages combined and colorcoded. all_langs.png

        Args:
            csv_file (str): path to csv file.

            lang_col (str, optional): Name of the csv column that contains lang. Defaults to "lang".

            mos_col (str, optional): Name of the csv column that contains MOS data. If None, MOS will not be returned. Defaults to None.

            eval_col (str, optional): Name of the csv column that contains PhonED data. If None, PhonED will not be returned. Defaults to None.
            
            img_dir (str, optional): Image directory. Defaults to "".
            
            title (str, optional): title. Defaults to None.
        """
        color_wheel = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        data_by_lang = cls.__load_by_lang(
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
        plt.tight_layout()
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
            plt.tight_layout()
            ax.figure.savefig(Path(img_dir).joinpath(f'{lang}.png'))
            ax.clear()

    @classmethod
    def hist_mos_by_lang(cls, csv_file, lang_col="lang", mos_col="mos_pred", img_dir="", title="{lang}", bins_count=50):
        """Create a histogram of MOS distribution for each language.

        Args:
            csv_file (str): path to csv file.

            lang_col (str, optional): Name of the csv column that contains lang. Defaults to "lang".

            mos_col (str, optional): Name of the csv column that contains MOS data. If None, MOS will not be returned. Defaults to None.

            img_dir (str, optional): Image directory. Defaults to "".

            title (str, optional): Title. You must specify {lang} in the title so the language can be formatted into it. 
            Defaults to "{lang}".

            bins_count (int, optional): Amount of histogram bins. Defaults to 50.
        """
        data = cls.__load_by_lang(csv_file, lang_col, mos_col)
        img_dir = Path(img_dir)
        
        for lang in data.keys():
            plt.hist(data[lang]['mos'], bins=bins_count)
            plt.title(title.format(lang=lang))
            plt.xlabel("Predicted MOS")
            plt.ylabel("n")
            plt.tight_layout()
            plt.savefig(img_dir.joinpath(f'{lang}_hist.png'))
            plt.close()

    @classmethod
    def analyse_track_asr(cls, track_path, predicted_text, original_text):
        """Analyze single wav file. Get it's MOS and PhonED.

        Args:
            track_path (str): Path to the wav file
            predicted_text (str): Text predicted by a model
            original_text (str): Reference text

        Returns:
            tuple[float, float]: MOS, PhonED
        """
        quality_metric = cls.eval_quality_single_wav(track_path)
        asr_metric = cls.eval_asr(predicted_text, original_text)
        return quality_metric, asr_metric
    
def test_model(model):
    evaluated_file = Path('refs/asr/evaluated.csv')
    model_author, model_name = model.split('/')

    #file handling
    models_dir = Path('models/')
    model_dir = models_dir.joinpath(model_name)
    if not (model_dir.is_dir()): model_dir.mkdir()
    img_dir = model_dir.joinpath('img')
    if not (img_dir.is_dir()): img_dir.mkdir()

    # moving evaluated.csv to model's directory
    if not model_dir.joinpath('evaluated.csv').is_file():
        os.rename(evaluated_file, model_dir.joinpath('evaluated.csv'))
    evaluated_file = model_dir.joinpath('evaluated.csv')
    evaluated_quality_file = model_dir.joinpath('out.csv')

    # creating PhonED histogram plot
    Evaluator.hist_phoned_by_model(model_name)

    # predicting MOS using NISQA model
    Evaluator.eval_quality_from_csv(
        csv_file=str(evaluated_file),
        path_col='new_path',
        save_to=evaluated_quality_file
    )
    
    # creating scatter plot MOS x PhonED for each lang
    Evaluator.plot_by_lang(
        model_dir.joinpath('out.csv'),
        img_dir=img_dir,
        title=f'{model} model'
    )

if __name__ == "__main__":
    test_model('speech31/wav2vec2-large-TIMIT-IPA')
