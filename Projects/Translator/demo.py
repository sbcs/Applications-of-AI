import queue
import sys
import time

import numpy as np
import sounddevice as sd
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Global variables for audio capture
SAMPLE_RATE = 16000
CHANNELS = 1  # Whisper expects mono audio

# A queue to hold audio data blocks
audio_queue = queue.Queue()

# Flag to control the recording/transcription loop
RUNNING = True

def audio_callback(indata, frames, time_info, status):
    """
    This callback is called by sounddevice whenever there's new audio data.
    We add the recorded frames to the queue to be processed later.
    """
    if status:
        print(status, file=sys.stderr)
    # Convert the recorded chunk to float32 NumPy
    # (sounddevice streams might come as int16, etc.)
    audio_data = indata.copy().flatten()
    audio_queue.put(audio_data)

def transcribe_chunk(model, processor, audio_chunk):
    """
    Given a raw audio chunk (NumPy array at 16kHz),
    run it through Whisper and return a transcription string.
    """
    import torch  # Make sure Torch is imported here or at the top
    # Convert to PyTorch tensor (1 x num_samples)
    audio_tensor = torchaudio.functional.resample(
        torch.from_numpy(audio_chunk), 
        orig_freq=SAMPLE_RATE, 
        new_freq=SAMPLE_RATE
    ).unsqueeze(0)

    # Extract features using the Whisper processor
    input_features = processor(
        audio_tensor.squeeze().numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).input_features

    # Generate predicted token IDs
    predicted_ids = model.generate(input_features)

    # Decode into text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def main():
    # Declare the global variable at the start
    global RUNNING

    import torch  # Import inside main to avoid any early import issues
    
    # Initialize Whisper model/processor
    model_name = "openai/whisper-base"  # or "openai/whisper-base", etc.
    print(f"Loading model '{model_name}'...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Set model to evaluation mode (just a best practice)
    model.eval()
    
    # Start recording from the default input device
    # The callback will feed audio data to the queue
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=16000,  # ~1 second of audio
        channels=CHANNELS,
        callback=audio_callback,
        dtype='float32'
    ):
        print("Recording and transcribing... Press Ctrl+C to stop.\n")
        
        # We’ll process audio in chunks as they come in from the queue
        # This loop checks the queue every half second
        try:
            while RUNNING:
                # If we have any data in the queue, transcribe it
                if not audio_queue.empty():
                    # Get next chunk from the queue
                    audio_chunk = audio_queue.get()
                    
                    # Transcribe that chunk
                    transcription = transcribe_chunk(model, processor, audio_chunk)
                    
                    # Print it out
                    print(">>", transcription)
                    
                # Sleep briefly so we’re not overloading the CPU
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping transcription...")
        finally:
            RUNNING = False

if __name__ == "__main__":
    main()
