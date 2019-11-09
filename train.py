import queue
import curses
import argparse

import os.path
import asyncio
import numpy as np
from model import *

sample_shape = (513, 25, 1)
weights_path = 'classifier_weights.h5'

audio_array = queue.Queue()
key_array = queue.Queue()
batch_size = 1
fft_window_size = 512

def main():
    model = TapModel(shape=sample_shape, weights_path=weights_path)

    curses.wrapper(start_console)

async def collector_task(model):
    if audio_array.size() >= batch_size:
        x = np.array([audioFileToSpectrogram(audio_array.get(), fft_window_size) for i in range(batch_size)])
        y = np.array([key_array.get() for i in range(batch_size)])

        model.train_on_spec_arr(
            specs = x,
            keys = y
            )

    await asyncio.sleep(.1)

# this will parse the last key press as audio
def cut_last_beat():
    # TODO setup pyaudio
    # use librosa to split into beats
    # cut last beat
    return None

# this will asynchronously collect keys & audio
def start_console(win):
    win.nodelay(True)
    key = ""
    win.clear()

    text = "Welcome! Start typing below, I'm listening\n\n"
    win.addstr(text)

    while True:
        try:
            key = win.getkey()
            if ord(key) == 127:
                text = text[:-1]
            else:
                text += key
            
            win.clear()
            win.addstr(text)
            if key == os.linesep:
                break
            
            key_array.put(key)
            audio_array.put(cut_last_beat())

        except Exception as e:
            pass

if __name__ == '__main__':
    main()
