## tiptap 
tiptap is a playful experimental package for converting audio of typing into a guess of what was actually typed.

Tiptap uses **pyaudio** along with **Soundflower** to capture a buffered stream of microphone data as raw amplitude data. Various methods from **librosa** are used to split the input audio along silent axes to isolate keyboard 'taps'. The training method correlates user typing to the console with input buffers captured from the microphone, allowing the model to learn in real-time.

## Training

    python train.py
   
   train.py will open a typing console and begin capturing microphone audio. You can then start typing and text you'd like. train.py will correlate your keystrokes to the audio captured, and feed this into the training model. Press **Enter** to abort training.

## Testing

    python test.py [-m model-weights]
  
  Once you've trained your model, you can run test.py to start guessing keystrokes. test.py will break to a new line and will begin capturing microphone audio. Type a sentence or a sequence of characters; when you press **Enter**, microphone audio will stop recording, the input audio will be analyzed, and the model will print out a guess of what letters were typed, along with error statistics.
