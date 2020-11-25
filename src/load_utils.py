import miniaudio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pathlib
import pandas as pd

pd.set_option('display.max_colwidth', 30)
pd.set_option('display.width', 500)





def tsv2df(lang="es", split="dev"):
    split = split + ".tsv"
    df = pd.read_table(datadir/lang/split, low_memory=False)
    df = df[["client_id", "path", "sentence", "locale"]]
    df["path"] = df.path.apply(lambda p: os.path.join(datadir/lang/"clips/", p))
    df["set"] = split.split(".")[0]
    df["target"] = df.locale.map(lang2target)
    print(df.shape)
    return df


def construct_metadf(langs, splits):
    dflist = [pd.concat([tsv2df(lang, split) for lang in langs])
          for split in splits]
    return pd.concat(dflist, ignore_index=True)

metadf = construct_metadf(langs, splits)



from itertools import combinations


def list_joint_speakers(df):
    """ find joint speakers between splits
    """
    splits = df.set.unique()
    joint = set()
    for combo in combinations(splits, 2):
        joint = joint.union(set(df[df["set"]==combo[0]].client_id.to_numpy()) &\
                    set(df[df["set"]==combo[1]].client_id.to_numpy()))
    return list(joint)
 
joint = list_joint_speakers(metadf)


tdf = metadf[~metadf.client_id.isin(joint)]


def test_disjunction(df):
    """ test disjunction of speakers
    """ 
    split2set = {split: set(df[df["set"]==split.split(".")[0]].client_id.to_numpy()) for split in splits}
    for split, spk in split2set.items():
        print("split {} has {} speakers".format(split, len(spk)))

    traintest = split2set["train"] & split2set["test"]
    traindev = split2set["train"] & split2set["dev"]
    devtest = split2set["dev"] & split2set["test"]
    if traintest != set():
        print(f"train and test have {len(traintest)} mutual speakers")
    if traindev  != set():
        print(f"train and dev have {len(traindev)} mutual speakers")
    if devtest != set():
        print(f"dev and test have {len(devtest)} mutual speakers")
    print(f"{len(traintest & devtest)} speakers are common to all sets")


test_disjunction(metadf)
test_disjunction(tdf)


tdf["dur"] = np.array([miniaudio.mp3_get_file_info(path).duration for path in tdf.path], np.float32)


tdf.head()


sns.set(rc={'figure.figsize': (8, 6)})
ax = sns.countplot(
    x="set",
    order=splits,
    hue="locale",
    hue_order=langs,
    data=tdf)
ax.set_title("# of audio samples")
plt.show()




def plot_duration_distribution(data):
    sns.set(rc={'figure.figsize': (8, 6)})
    
    ax = sns.boxplot(
        x="set",
        order=splits,
        y="dur",
        hue="locale",
        hue_order=langs,
        data=data)
    ax.set_title("Median audio file duration in seconds")
    plt.show()

    ax = sns.barplot(
        x="set",
        order=splits,
        y="dur",
        hue="locale",
        hue_order=langs,
        data=data,
        ci=None,
        estimator=np.sum)
    ax.set_title("Total amount of audio in seconds")
    plt.show()


plot_duration_distribution(tdf)



groupby_lang = tdf[["locale", "dur"]].groupby("locale")
    
total_dur = groupby_lang.sum()
target_lang = total_dur.idxmax()[0]
print("target lang:", target_lang)
print("total durations:")
display(total_dur)


