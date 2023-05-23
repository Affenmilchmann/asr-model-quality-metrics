## Measuring quality of testing data

First, install NISQA model into */refs/nisqa/* directory. Follow their instructions from here: https://github.com/gabrielmittag/NISQA

In order to check yourself, make sure that model weights are available at *refs/nisqa/weights/nisqa_mos_only.tar*

### The main class
The main class is *DataQuality.py*. It is a static class.

It has several methods (see docstrings for details):
 * eval_quality_single_wav() - evaluates a single wav file's quality by predicting it's MOS.
 * eval_quality_from_csv() - evaluates multiple wav files 
 * eval_asr() - evaluate Phonetic Edit Distance of recognized and reference texts.
 * statistic_by_lang() - get statistic of MOS and PhonED correlation by lang.
 * hist_phoned_by_model() - form a plot image of segments distribution by PhonED. 
 * plot_by_lang() - form a scatter plot (MOS x PhonED) for each lang
 * hist_mos_by_lang() - form a plot image of segments distribution by mos by lang.