

import matplotlib.pyplot as plt
import lidbox.models.xvector as xvector

import numpy as np
import tensorflow as tf
import os
import lidbox.data.steps as ds_steps
import pandas as pd
from src.data import load_metadf, splits, lang2target, target2lang
from src.features import pipeline_from_metadata, assert_finite
from src.model import evaluate
from lidbox.util import classification_report
from lidbox.visualize import draw_confusion_matrix
from pprint import pprint

import miniaudio
import scipy
import sounddevice as sd



meta = load_metadf()
    # mapping from dataset split names to feature tf.data.Dataset
split2ds = {
    split: pipeline_from_metadata(meta[meta["split"]==split], shuffle=split=="train")
    for split in splits
}

test_ds = split2ds["test"].map(lambda x: dict(x, input=x["logmelspec"])).batch(1)
test_meta = evaluate(test_ds, meta)


#  extract_audio
import time
import subprocess
def extract_audio(filename):
    t = time.time()
    output_filename = filename.split(".")[0]+".mp3"
    command = f"ffmpeg -i {filename} -ab 160k -ac 1 -ar 16000 -vn {output_filename}"
    subprocess.call(command, shell=True)
    print(time.time()-t)



extract_audio("Euronews_fr.mp4")



# animate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.features import cmvn,read_mp3,plot_signal,remove_silence, logmelspectrograms, plot_spectrogram
import librosa
import scipy
from src.model import create_model, cachedir


filename = 'fr_example_china.mp3'


model = create_model(
        num_freq_bins=40,
        num_labels=len(lang2target))
_ = model.load_weights(os.path.join(cachedir, "model", model.name))

def sig2logspec(signal):
    logmelspec = logmelspectrograms(signal.reshape(1,-1), 16000)
    logmelspec_smn = cmvn(logmelspec)
    return logmelspec_smn.numpy()[0]


def generate_demo_vid(filename, model=model, window = 16000*4, jump = 512):
    sound, rate = librosa.load(filename,sr=None)
    print(rate)
    print(len(sound))
    print((len(sound)-window)//jump)
    fig, axs = plt.subplots(2)
    # axs[0].grid(False)
    # axs[1].grid(False)
    axs[1].set_yticklabels([])
    axs[1].set_xticklabels([])
    axs[0].set_yticklabels([])
    axs[0].set_xticklabels([])
    fig.set_figheight(3)
    fig.set_figwidth(10)

    ims = []
    print("start loop")
    for i in range((len(sound)-window)//jump):
        arr = sig2logspec(sound[i*jump:(i*jump)+window]).T
        im0, = axs[0].plot(sound[i*jump:(i*jump)+window], color="mediumblue", linewidth=.5)
        logit = model(np.expand_dims(arr.T,0), training=False).numpy()
        idx = logit.argmax()
        probs = scipy.special.softmax(logit[0])
        pred = target2lang[idx]
        im1 = axs[1].imshow(arr, cmap="viridis", animated=True)
        title = axs[1].text(0.5,0.2,f"{pred} {probs[idx]:.2%}", 
                        size=plt.rcParams["axes.titlesize"],
                        color='w', fontsize=40,
                        ha="center", transform=axs[1].transAxes, )
        ims.append([im0, im1, title])
    print("end loop")
    
    ani = animation.ArtistAnimation(fig, ims, interval=32, blit=False, repeat=False)
    ani.save(filename.split(".")[0]+"_spec.mp4")
    # plt.show()




    
generate_demo_vid("Euronews_fr.mp3")
generate_demo_vid("Euronews_de.mp3")    
generate_demo_vid("Euronews_es.mp3")



#MIC STREAM
import pyaudio
import numpy as np
import sounddevice as sd
import audioop

def play(signal, fs):
    sd.play(signal, fs)


FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
CHUNK = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "file.wav"
 
# start Recording
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print("recording...")
frames = []
for i in range(0, int(RATE / CHUNK * 20)):
    data = stream.read(CHUNK)
    vol = audioop.rms(data, 2)
    data = np.frombuffer(data, dtype=np.float32)
    frames.append(data)
    arr = sig2logspec(np.hstack(frames[-3:])).T
    logit = model(np.expand_dims(arr.T,0), training=False).numpy()
    idx = logit.argmax()
    probs = scipy.special.softmax(logit[0])
    pred = target2lang[idx]
    print(f"{pred} {probs[idx]:.2%} {vol}")


print("finished recording")
stream.stop_stream()
stream.close()
audio.terminate()







import pyaudio
from urllib.request import urlopen
import numpy as np
import sounddevice as sd
import scipy.signal
url = "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_700KB.mp3"
signal, resample_rate = librosa.load("file_example_MP3_700KB.mp3")
u = urlopen(url)

import miniaudio


import librosa
from src.features import plot_signal
import matplotlib.pyplot as plt
plot_signal(signal)
plt.show()
signal.max()

f = miniaudio.mp3_read_file_f32("file_example_MP3_700KB.mp3")
f.sample_rate
f.nchannels

f.samples

play(signal, 16000)
sd.stop()
len(signal)
873216/32000

pyaud = pyaudio.PyAudio()
srate=32000
stream = pyaud.open(format = pyaud.get_format_from_width(1),
                channels = 2,
                rate = srate,
                output = True)


data = u.read(10000)
data = np.frombuffer(data, dtype=np.float32)
play(data, srate)

len(data)


while data:
    stream.write(data)
    data = u.read(8192)

def play(signal, fs):
    sd.play(signal, fs)

