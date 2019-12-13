## tiptap 
tiptap is a playful experimental package for converting audio of typing into a guess of what was actually typed.

Tiptap uses **pyaudio** to capture a buffered stream of microphone data as raw amplitude data. Various methods from **librosa** are used to split the input audio along silent axes to isolate keyboard 'taps'. The training method correlates user typing to the console with input buffers captured from the microphone, allowing the model to learn in real-time.

## Setup
    If you're running on MacOSx:

    # ./setup.sh

## Training

    # sudo python run.py
   
   Training will accept text and begin capturing microphone audio. You can then start typing and text you'd like. train.py will correlate each keystroke to the last 2 seconds of audio captured, and feed this into the training model. Press **Esc** to abort training.

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
