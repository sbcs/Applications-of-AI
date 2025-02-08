import queue
import sys
import time
import select

import sounddevice as sd
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Global variables for audio capture
SAMPLE_RATE = 16000
CHANNELS = 1  # Whisper expects mono audio
VOLUME_THRESHOLD = 0.01  # Adjust this threshold to detect when speaking

# A queue to hold audio data blocks
audio_queue = queue.Queue()

# Flag to control the recording/transcription loop
RUNNING = True

def audio_callback(indata, frames, time_info, status):
    """
    This callback is called by sounddevice whenever there's new audio data.
    We add the recorded frames to the queue to be processed later if volume exceeds threshold.
    """
    if status:
        print(status, file=sys.stderr)

    # Check if the audio level exceeds the threshold
    volume_norm = (indata ** 2).mean() ** 0.5
    if volume_norm > VOLUME_THRESHOLD:
        audio_data = indata.flatten().tolist()
        audio_queue.put(audio_data)

def transcribe_chunk(model, processor, audio_chunk):
    """
    Given a raw audio chunk (list of floats at 16kHz),
    run it through Whisper and return a transcription string.
    """
    import torch

    # Convert to PyTorch tensor (1 x num_samples)
    audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0)

    # Resample audio to the expected SAMPLE_RATE if needed
    if audio_tensor.shape[1] != SAMPLE_RATE:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=SAMPLE_RATE, new_freq=SAMPLE_RATE
        )

    # Extract features using the Whisper processor
    input_features = processor(
        audio_tensor.squeeze().numpy(),  # This uses NumPy internally
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).input_features

    # Generate predicted token IDs with forced language as English
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe") 
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    # Decode into text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def main():
    global RUNNING
    import torch

    # Initialize Whisper model/processor
    model_name = "openai/whisper-base"
    print(f"Loading model '{model_name}'...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    while True:
        user_input = input("Would you like to speak? Press Y to start or type 'no' to stop: ").strip().lower()
        if user_input == 'y':
            RUNNING = True
            print("Recording and transcribing... Type 'no' to stop.")
            
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=16000,  # ~1 second of audio
                channels=CHANNELS,
                callback=audio_callback,
                dtype='float32'
            ):
                try:
                    while RUNNING:
                        if not audio_queue.empty():
                            audio_chunk = audio_queue.get()
                            transcription = transcribe_chunk(model, processor, audio_chunk)
                            print(">>", transcription)
                        
                        # Check if the user wants to stop
                        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                            stop_input = sys.stdin.readline().strip().lower()
                            if stop_input == 'no':
                                RUNNING = False

                        time.sleep(0.1)
                except KeyboardInterrupt:
                    print("\nStopping transcription...")
                finally:
                    RUNNING = False
        elif user_input == 'no':
            print("Exiting program.")
            break
        else:
            print("Invalid input. Please type 'Y' to start or 'no' to stop.")

if __name__ == "__main__":
    main()
