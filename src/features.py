#!/usr/bin/env python3
# coding: utf-8
######################################
# authors                            #
# marcalph <marcalph@protonmail.com> #
######################################

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations
import os
import scipy.signal
import miniaudio
import sounddevice as sd
import tensorflow as tf
from lidbox.features.audio import framewise_rms_energy_vad_decisions
import matplotlib.patches as patches
import numpy as np

from lidbox.features import cmvn
from lidbox.features.audio import spectrograms
from lidbox.features.audio import linear_to_mel

import lidbox.data.steps as ds_steps
import lidbox.data

from src.data import load_metadf
from src.data import langs, splits, lang2target, target2lang


TF_AUTOTUNE = tf.data.experimental.AUTOTUNE

def read_mp3(path, resample_rate=16000):
    if isinstance(path, bytes):
        # If path is a tf.string tensor, it will be in bytes
        path = path.decode("utf-8")
        
    f = miniaudio.mp3_read_file_f32(path)
    # Downsample to target rate, 16 kHz is commonly used for speech data
    new_len = round(len(f.samples) * float(resample_rate) / f.sample_rate)
    signal = scipy.signal.resample(f.samples, new_len)
    
    # Normalize to [-1, 1]
    signal /= np.abs(signal).max()
    return signal, resample_rate


def play(signal, fs):
    sd.play(signal, fs)

    
def plot_signal(data, figsize=(6, 0.5), **kwargs):
    ax = sns.lineplot(data=data, lw=0.1, **kwargs)
    ax.set_axis_off()
    ax.margins(0)
    plt.gcf().set_size_inches(*figsize)
    # plt.show()

    
def remove_silence(signal, rate):
    window_ms = tf.constant(10, tf.int32)
    window_frames = (window_ms * rate) // 1000
    
    # Get binary VAD decisions for each 10 ms window
    vad_1 = framewise_rms_energy_vad_decisions(
        signal=signal,
        sample_rate=rate,
        frame_step_ms=window_ms,
        # Do not return VAD = 0 decisions for sequences shorter than 300 ms
        min_non_speech_ms=300,
        strength=0.1)
    
    # Partition the signal into 10 ms windows to match the VAD decisions
    windows = tf.signal.frame(signal, window_frames, window_frames)
    # Filter signal with VAD decision == 1
    return tf.reshape(windows[vad_1], [-1])


def logmelspectrograms(signals, rate):
    powspecs = spectrograms(signals, rate)
    melspecs = linear_to_mel(powspecs, rate, num_mel_bins=40)
    return tf.math.log(melspecs + 1e-6)

# load samples
def metadata_to_dataset_input(meta):   
    # Create a mapping from column names to all values under the column as tensors
    return {
        "id": tf.constant(meta.index.astype(str), tf.string),
        "path": tf.constant(meta.path, tf.string),
        "lang": tf.constant(meta.locale, tf.string),
        "target": tf.constant(meta.target, tf.int32),
        "split": tf.constant(meta.split, tf.string),
    }


def read_mp3_wrapper(x):
    signal, sample_rate = tf.numpy_function(
        # Function
        read_mp3,
        # Argument list
        [x["path"]],
        # Return value types
        [tf.float32, tf.int64])
    return dict(x, signal=signal, sample_rate=tf.cast(sample_rate, tf.int32))


def remove_silence_wrapper(x):
    return dict(x, signal=remove_silence(x["signal"], x["sample_rate"]))


def batch_extract_features(x):
    with tf.device("GPU"):
        signals, rates = x["signal"], x["sample_rate"]
        logmelspecs = logmelspectrograms(signals, rates[0])
        logmelspecs_smn = cmvn(logmelspecs)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(logmelspecs)
        mfccs = mfccs[...,1:21]
        mfccs_cmvn = cmvn(mfccs)
    return dict(x, logmelspec=logmelspecs_smn, mfcc=mfccs_cmvn)


def signal_is_not_empty(x):
    return tf.size(x["signal"]) > 0
    

def pipeline_from_metadata(data, shuffle=False):
    if shuffle:
        # Shuffle metadata to get an even distribution of labels
        data = data.sample(frac=1, random_state=42)
    ds = (
        # Initialize dataset from metadata
        tf.data.Dataset.from_tensor_slices(metadata_to_dataset_input(data))
        # Read mp3 files from disk in parallel
        .map(read_mp3_wrapper, num_parallel_calls=TF_AUTOTUNE)
        # Apply RMS VAD to drop silence from all signals
        .map(remove_silence_wrapper, num_parallel_calls=TF_AUTOTUNE)
        # Drop signals that VAD removed completely
        .filter(signal_is_not_empty)
        # Extract features in parallel
        .batch(1)
        .map(batch_extract_features, num_parallel_calls=TF_AUTOTUNE)
        .unbatch()
    )
    return ds



def plot_spectrogram(S, cmap="viridis", figsize=None, **kwargs):
    if figsize is None:
        figsize = S.shape[0]/50, S.shape[1]/50
    ax = sns.heatmap(S.T, cbar=False, cmap=cmap, **kwargs)
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.margins(0)
    plt.gcf().set_size_inches(*figsize)
    # plt.show()

def plot_cepstra(X, figsize=None):
    if not figsize:
        figsize = (X.shape[0]/50, X.shape[1]/20)
    plot_spectrogram(X, cmap="RdBu_r", figsize=figsize)



def assert_finite(x, y):
    tf.debugging.assert_all_finite(x, "non-finite input")
    tf.debugging.assert_non_negative(y, "negative target")
    return x, y


if __name__ == "__main__":
    meta = load_metadf()
    # Mapping from dataset split names to tf.data.Dataset objects
    split2ds = {
        split: pipeline_from_metadata(meta[meta["split"]==split], shuffle=split=="train")
        for split in splits
    }
