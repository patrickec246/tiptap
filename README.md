## tiptap 
tiptap learns audio correlations with keystrokes during training, then attempts to guess keystrokes from raw audio input during the testing phase.

Tiptap uses **pyaudio** to capture a buffered stream of microphone data as raw amplitude data. Audio is then processed as a specotrogram and passed through a convolutional model which maps the 2d spectrogram array to a [**batch_num** x key_classes] keystrokes array representing keystroke guesses from the buffer.

## Setup
    If you're running on MacOSx:

    # ./setup.sh

## Training

    # sudo python run.py
   
   Training will accept text and begin capturing microphone audio. You can then start typing and text you'd like. train.py will correlate each keystroke to the last 2 seconds of audio captured, and feed this into the training model. Press **Esc** to abort training.
   
   Training occurs as fast as your gpu/cpu can process the datapoints which are sampled from the audio buffer, over time providing a random distribution over a sliding window of the audio buffer.


   ```
   (env) (base) local:tiptap local$ sudo python train.py
    Using plaidml.keras.backend backend.
    + ------------------------------- +
    | Start typing, I am listening :) |
    + ------------------------------- +
    Datapoints generated: 1305
    You pressed "ESC", training complete.
   ```

## Testing

    # python run.py -test
  
  Once you've trained your model, you can run with [-test] to start guessing keystrokes. Testing will break to a new line and will begin capturing microphone audio. Type a sentence or a sequence of characters; when you press **Enter**, microphone audio will stop recording, the input audio will be analyzed, and the model will print out a guess of what letters were typed, along with error.

  ```
  (env) (base) local:tiptap local$ sudo python train.py -test
  Using plaidml.keras.backend backend.

  Enter some text: Hello
  Based on audio, I think you typed: ga3ggtlt (accuracy: 0%)

  ---

  (env) (base) local:tiptap local$ sudo python train.py -test
  Using plaidml.keras.backend backend.

  Enter some text: madman3
  
  Based on audio, I think you typed: naemap3 (accuracy: 57%)
  ```
