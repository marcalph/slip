import os
import pathlib
import pandas as pd
pd.set_option('display.max_colwidth', 30)
pd.set_option('display.width', 500)

languages = """
    es
    fr
    de
""".split()

splits = """
    train
    test
    dev
""".split()

datadir = pathlib.Path("./common-voice/cv-corpus")

langs = tuple(sorted(languages))
lang2target = {lang: target for target, lang in enumerate(langs)}

print("lang2target:", lang2target)
print("langs:", langs)


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

test_disjunction(metadf)
test_disjunction(tdf)

