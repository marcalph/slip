#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################

import os

def test_disjunction(df):
    """ test disjunction of speakers
    """ 
    split2set = {split: set(df[df["split"]==split.split(".")[0]].client_id.to_numpy()) for split in splits}
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


# check audio exist
def test_file_presence(df):
    for _, row in df.iterrows():
        assert os.path.exists(row["path"]), row["path"] + " does not exist"
    print("ok")

