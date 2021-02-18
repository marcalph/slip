#!/usr/bin/env python3
# coding: utf-8
######################################
# authors                            #
# marcalph <marcalph@protonmail.com> #
######################################

# boring imports
import miniaudio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pathlib
import pandas as pd
from itertools import combinations

pd.set_option('display.max_colwidth', 30)
pd.set_option('display.width', 500)


#boring constants
languages = """
    es
    fr
    de
    ar
    ru
""".split()

splits = """
    train
    test
    dev
""".split()

datadir = pathlib.Path("./common-voice/cv-corpus")
langs = tuple(sorted(languages))
lang2target = {lang: target for target, lang in enumerate(langs)}
target2lang = {target: lang for target, lang in enumerate(langs)}



def tsv2df(datadir, lang, split):
    """create dataframe
    """
    split = split + ".tsv"
    df = pd.read_table(datadir/lang/split, low_memory=False)
    df = df[["client_id", "path", "sentence", "locale"]]
    df["path"] = df.path.apply(lambda p: os.path.join(datadir/lang/"clips/", p))
    df["split"] = split.split(".")[0]
    print(split)
    if split=="train.tsv":
        df = df.sample(min(30000, len(df)))
    print(df.shape)
    return df


def construct_metadf(datadir, langs, splits):
    dflist = [pd.concat([tsv2df(datadir, lang, split) for lang in langs])
          for split in splits]
    df = pd.concat(dflist, ignore_index=True)
    language_map = {lang: target for target, lang in enumerate(langs)}
    df["target"] = df.locale.map(language_map)
    return df


def list_joint_speakers(df):
    """ find joint speakers between splits
    """
    splits = df.split.unique()
    joint = set()
    for combo in combinations(splits, 2):
        joint = joint.union(set(df[df["split"]==combo[0]].client_id.to_numpy()) &\
                    set(df[df["split"]==combo[1]].client_id.to_numpy()))
    return list(joint)



def plot_duration_distribution(data):
    _, axes = plt.subplots(1, 3)

    sns.set(rc={'figure.figsize': (6, 8)})
    sns.countplot(
        x="split",
        order=splits,
        hue="locale",
        hue_order=langs,
        data=data,
        ax=axes[0]).set_title("# of audio samples")

    sns.boxplot(
        x="split",
        order=splits,
        y="dur",
        hue="locale",
        hue_order=langs,
        data=data,
        ax=axes[1]).set_title("Median duration")

    sns.barplot(
        x="split",
        order=splits,
        y="dur",
        hue="locale",
        hue_order=langs,
        data=data,
        ci=None,
        estimator=np.sum,
        ax=axes[2]).set_title("Audio amount")
    plt.show()




def oversampling(meta):
    groupby_lang = meta[["locale", "dur"]].groupby("locale")
    
    total_dur = groupby_lang.sum()
    target_lang = total_dur.idxmax()[0]
    print("target lang:", target_lang)
    print("total durations:")
    print(total_dur)
    
    total_dur_delta = total_dur.loc[target_lang] - total_dur
    print("total duration delta to target lang:")
    print(total_dur_delta)
    
    median_dur = groupby_lang.median()
    print("median durations:")
    print(median_dur)
    
    sample_sizes = (total_dur_delta / median_dur).astype(np.int32)
    print("median duration weighted sample sizes based on total duration differences:")
    print(sample_sizes)
    
    samples = []
    
    for lang in groupby_lang.groups:
        sample_size = sample_sizes.loc[lang][0]
        sample = (meta[meta["locale"]==lang]
                  .sample(n=sample_size, replace=True, random_state=42)
                  .reset_index())
        samples.append(sample)

    return pd.concat(samples)



def load_metadf():
    meta = construct_metadf(datadir, langs, splits)
    # remove joint speakers
    joint = list_joint_speakers(meta)
    meta = meta[~meta.client_id.isin(joint)]
    # add duration
    meta["dur"] = np.array([miniaudio.mp3_get_file_info(path).duration for path in meta.path], np.float32)
    meta = pd.concat([oversampling(meta[meta["split"]=="train"]), meta]).sort_index()
    meta = meta.drop(["index"], axis=1)
    meta = meta.dropna()
    return meta








if __name__ == "__main__":
    meta = load_metadf()
    assert not meta.isna().any(axis=None), "NaNs in metadata after augmentation"
    plot_duration_distribution(meta)
