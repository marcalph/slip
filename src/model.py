#!/usr/bin/env python3
# coding: utf-8
######################################
# authors                            #
# marcalph <marcalph@protonmail.com> #
######################################

import matplotlib.pyplot as plt
import lidbox.models.xvector as xvector

import numpy as np
import tensorflow as tf
import os
import lidbox.data.steps as ds_steps
import pandas as pd
import miniaudio
from src.data import load_metadf, splits, lang2target, target2lang
from src.features import pipeline_from_metadata, assert_finite

from lidbox.util import classification_report
from lidbox.visualize import draw_confusion_matrix
from pprint import pprint



TF_AUTOTUNE = tf.data.experimental.AUTOTUNE

workdir = "./work"
cachedir = os.path.join(workdir, "cache")


def as_model_input(x, model_input_type="logmelspec"):
    return x[model_input_type], x["target"]


def create_model(num_freq_bins, num_labels):
    model = xvector.create([None, num_freq_bins], num_labels, channel_dropout_rate=0.8)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5))
    return model


# # caching for faster consumption
# _ = ds_steps.consume(split2ds["train"], log_interval=2000)
# os.makedirs(os.path.join(cachedir, "data"))
# split2ds["train"] = split2ds["train"].cache(os.path.join(cachedir, "data", "train"))
# _ = ds_steps.consume(split2ds["train"], log_interval=2000)


def train(model_input_type="logmelspec"):
    _ = ds_steps.consume(split2ds["train"].map(as_model_input).map(assert_finite), log_interval=5000)
    model = create_model(
        num_freq_bins=20 if model_input_type == "mfcc" else 40,
        num_labels=len(lang2target))
    model.summary()
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(cachedir, "tensorboard", model.name),
            update_freq="epoch",
            write_images=True,
            profile_batch=0,
        ),
        # earlystopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
        ),
        # ckpt
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(cachedir, "model_aug", model.name),
            monitor='val_loss',
            save_weights_only=True,
            save_best_only=True,
            verbose=1,
        ),
    ]
    train_ds = split2ds["train"].map(as_model_input).shuffle(1000)
    dev_ds = split2ds["dev"].cache(os.path.join(cachedir, "data", "dev")).map(as_model_input)
    history = model.fit(
        train_ds.batch(1),
        validation_data=dev_ds.batch(1),
        callbacks=callbacks,
        verbose=2,
        epochs=100)


def predictions_to_dataframe(ids, predictions):
    return (pd.DataFrame.from_dict({"id": ids, "logit": predictions})
            .set_index("id", drop=True, verify_integrity=True))


def predict_with_model(model, ds, predict_fn=None):
    """
    Map callable model over all batches in ds, predicting values for each element at key 'input'.
    """
    if predict_fn is None:
        def predict_fn(x):
            with tf.device("GPU"):
                return x["id"], model(x["input"], training=False)

    ids = []
    predictions = []
    for id, pred in ds.map(predict_fn, num_parallel_calls=TF_AUTOTUNE).unbatch().as_numpy_iterator():
        ids.append(id.decode("utf-8"))
        predictions.append(pred)
    return predictions_to_dataframe(ids, predictions)



def evaluate(test_ds, meta, model_input_type="logmelspec"):
    model = create_model(
        num_freq_bins=40, #20 if model_input_type == "mfcc" else 
        num_labels=len(lang2target))
    _ = model.load_weights(os.path.join(cachedir, "model_aug", model.name))
    utt2pred = predict_with_model(model, test_ds)
    #remove index name
    utt2pred.index.name= None
    test_meta = meta[meta["split"]=="test"]
    test_meta.index = test_meta.index.astype(int)
    utt2pred.index = utt2pred.index.astype(int)
    test_meta = test_meta.join(utt2pred)

    true_sparse = test_meta.target.to_numpy(np.int32)
    pred_dense = np.stack(test_meta.logit)
    test_meta["pred"] = [target2lang[idx] for idx in pred_dense.argmax(axis=1).astype(np.int32)]
    report = classification_report(true_sparse, pred_dense, lang2target)
    pprint(report)
    for m in ("avg_detection_cost", "avg_equal_error_rate", "accuracy"):
        print("{}: {:.3f}".format(m, report[m]))
    
    lang_metrics = pd.DataFrame.from_dict({k: v for k, v in report.items() if k in lang2target})
    lang_metrics["mean"] = lang_metrics.mean(axis=1)
    fig, ax = draw_confusion_matrix(report["confusion_matrix"], lang2target)
    plt.show()
    return test_meta


import scipy


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









if __name__ == "__main__":
    meta = load_metadf()
    # mapping from dataset split names to feature tf.data.Dataset
    split2ds = {
        split: pipeline_from_metadata(meta[meta["split"]==split], shuffle=split=="train")
        for split in splits
    }
    test_ds = split2ds["test"].map(lambda x: dict(x, input=x["logmelspec"])).batch(1)
    test_meta = evaluate(test_ds, meta)

