import queue
import curses
import argparse
import pyaudio
import time
import librosa, librosa.display
import collections
import logging
import shutil
import keyboard
import threading
import math
import sys
import string
import traceback

import os.path
import asyncio
import numpy as np
import matplotlib.pyplot as plt

from model import *

logger = logging.getLogger(__file__)
handler = logging.FileHandler('output.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)

# general settings
window_threshold = 0.5

# keras settings
batch_size = 8
weights_path = 'classifier_weights.h5'
sample_shape = (513, 25, 1)

# librosa settings
fft_window_size = 1024
sampling_rate = 22050
channels = 1
sample_length_sec = 2

# pyaudio settings
pyaudio_format = pyaudio.paFloat32
buffer_size = 512

last_key = 0

audio_buffer = np.zeros((sampling_rate*sample_length_sec), dtype=np.float32)
keypress_times = []
keyrelease_times = []

def pyaudio_callback(input_data, frame_count, time_info, status):
    global audio_buffer

    audio_data = np.frombuffer(input_data, dtype=np.float32)
    audio_data = librosa.resample(audio_data, 2*sampling_rate, sampling_rate)
    audio_buffer = np.concatenate((audio_buffer[audio_data.shape[0]:], audio_data))

    return (input_data, pyaudio.paContinue)

# pyaudio stream setup
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio_format,
            rate=2*sampling_rate,
            channels=channels,
            frames_per_buffer=buffer_size,
            input=True,
            stream_callback=pyaudio_callback)

#keras model setup
model = TapModel(batch_num=batch_size)

def sanitize_key(name):
    name = name.strip()

    if 'space' in name:
        return ' '
    if name == 'enter':
        return '\n'
    if name in detectable_keys:
        return name
    elif 'shift' in name:
        return '$'
    return None

def round_to_interval(value, interval, dir='down'):
    if dir is 'down':
        return math.floor(value/interval) * interval
    else:
        return math.ceil(value/interval) * interval

def generate_key_array():
    global keypress_times, keyrelease_times

    if len(keypress_times) != len(keyrelease_times):
        return None

    return [(keypress_times[i][1], keyrelease_times[i][1], keypress_times[i][0]) for i in range(len(keypress_times))]


def generate_datapoint(sound_arr=None, key_array=None, batch_num=1, sampling_rate=22050):
    if key_array is None:
        return (None, None)

    # x
    spec = np.log1p(np.abs(librosa.stft(sound_arr, n_fft=fft_window_size)))
    if np.min(spec) == np.max(spec):
        return (None, None)
    img = np.clip((spec - np.min(spec)) / (np.max(spec) - np.min(spec)), 0, 1)

    # y
    output = np.zeros(shape=(batch_num, len(detectable_keys)))
    time_len = librosa.core.samples_to_time(sound_arr.shape[0])
    batch_time_len = time_len / batch_num
    min_time = time.time() - time_len

    for start_time, end_time, key in key_array:
        start_time, end_time = start_time - min_time, end_time - min_time
        if key is None or start_time < 0 or end_time < 0:
            continue 

        start_interval = round_to_interval(start_time, batch_time_len)
        end_interval = round_to_interval(end_time, batch_time_len, dir='up')
        for marker in np.arange(start_interval, end_interval, batch_time_len):
            bucket = int(marker / batch_time_len)
    
            denominator = batch_time_len
            numerator = min(end_time, marker + batch_time_len) - max(start_time, marker)
            ratio = numerator / denominator

            if ratio >= window_threshold:
                key_idx = detectable_keys.index(sanitize_key(key))
                output[bucket][key_idx] = 1.0

    return (img, output)

training = True
def train_func():
    datapoints = 0
    print('+ ------------------------------- +')
    print('| Start typing, I am listening :) |')
    print('+ ------------------------------- +')
    while training:
        try:
            keys = generate_key_array()
            x, y = generate_datapoint(audio_buffer, key_array=keys, batch_num=batch_size)
            if x is not None and y is not None:
                a, c = model.train_on_spec(x, y)
                datapoints += 1
                print('Datapoints generated: {}'.format(datapoints), end='\r')
                sys.stdout.flush()

                if datapoints % 10 == 0:
                    model.model.save_weights(weights_path)
        except IndexError as e:
            pass
        except Exception as ee:
            track = traceback.format_exc()
            return
    
    print('Datapoints generated: {}'.format(datapoints))
    print('You pressed "ESC", training complete.')

thread = threading.Thread(target=train_func)
keylock = threading.Lock()
keydowns = {}
keypresses = []

def key_event(e):
    global training, keypresses, keydowns, keylock

    key = sanitize_key(e.name)

    os.system('stty -echo')

    if e.name == 'esc':
        training = False
    elif key is None:
        return
    elif e.event_type is 'down':
        keypress_times.append((key, e.time))
        with keylock:
            keydowns[key] = e.time
    elif e.event_type is 'up':
        keyrelease_times.append((key, e.time))
        with keylock:
            keypush_time = keydowns[key]
            del keydowns[key]
            keypresses.append((keypush_time, e.time, key))

def train():
    keyboard.hook(key_event)
    thread.start()
    thread.join()


def test():
    input_d = input('\nEnter some text: ')

    spec = np.log1p(np.abs(librosa.stft(audio_buffer, n_fft=fft_window_size)))
    if np.min(spec) == np.max(spec):
        return
    img = np.expand_dims(np.clip((spec - np.min(spec)) / (np.max(spec) - np.min(spec)), 0, 1), 2)

    prediction = ''.join(model.predict_keys(img))

    acc, total = 0, 0
    for i in range(len(prediction)):
        total += 1
        if i < len(input_d) and input_d[i] == prediction[i]:
            acc += 1
    acc = round(100 * (acc / total))

    print('Based on audio, I think you typed: {} (accuracy: {}%)'.format(prediction, acc))

if __name__ == '__main__':
    args = sys.argv

    if len(args) >= 2:
        test()
    else:
        train()
        os.system('stty echo')
