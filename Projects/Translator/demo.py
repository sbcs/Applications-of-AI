import queue
import sys
import time
import sounddevice as sd
import torchaudio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    M2M100ForConditionalGeneration,
)

# Global variables for audio capture
SAMPLE_RATE = 16000
CHANNELS = 1  # Whisper expects mono audio
VOLUME_THRESHOLD = 0.015  # Adjust this threshold to detect when speaking

# A queue to hold audio data blocks
audio_queue = queue.Queue()

# Flag to control the recording/transcription loop
RUNNING = True

def audio_callback(indata, frames, time_info, status):
    """
    This callback is called by sounddevice whenever there's new audio data.
    If the volume exceeds a threshold, the audio block is added to a queue.
    """
    if status:
        print(status, file=sys.stderr)
    volume_norm = (indata ** 2).mean() ** 0.5
    if volume_norm > VOLUME_THRESHOLD:
        audio_data = indata.flatten().tolist()
        audio_queue.put(audio_data)

def transcribe_chunk(model, processor, audio_chunk, language="en"):
    """
    Given a raw audio chunk (list of floats at 16kHz), run it through Whisper and return
    the transcription as a string. The 'language' parameter forces the output language.
    """
    import torch

    # Convert to a PyTorch tensor (shape: 1 x num_samples)
    audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0)

    # (Optional) Resample if necessary. In this example, we assume SAMPLE_RATE.
    if audio_tensor.shape[1] != SAMPLE_RATE:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=SAMPLE_RATE, new_freq=SAMPLE_RATE
        )

    # Process the raw audio with the Whisper processor.
    input_features = processor(
        audio_tensor.squeeze().numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).input_features

    # Get forced decoder IDs using the provided language.
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    # Decode into text.
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def main():
    import torch

    # Load the Whisper model and processor for transcription.
    whisper_model_name = "openai/whisper-base"
    print(f"Loading transcription model '{whisper_model_name}'...")
    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
    model.eval()

    # Load the M2M100 translation model and tokenizer.
    translation_model_name = "facebook/m2m100_418M"
    print(f"Loading translation model '{translation_model_name}'...")
    translation_model = M2M100ForConditionalGeneration.from_pretrained(translation_model_name)
    translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
    translation_model.eval()

    # Mapping for target languages for M2M100.
    # M2M100 expects ISO language codes like "en", "es", "fr", etc.
    m2m100_language_codes = {
        "1": "en",  # English
        "2": "es",  # Spanish
        "3": "fr",  # French
        "4": "de",  # German
        "5": "it",  # Italian
        "6": "pt"   # Portuguese
    }
    language_names = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese"
    }

    while True:
        user_input = input("Would you like to speak? Press Y to start or type 'no' to stop: ").strip().lower()
        if user_input == 'y':
            global RUNNING
            RUNNING = True
            print("Recording and transcribing live... Press Ctrl+C to stop.")

            # List to accumulate live transcriptions (raw text).
            transcribed_chunks = []
            # (Optional) Also accumulate raw audio if needed.
            all_audio = []

            # Clear the audio queue before starting.
            while not audio_queue.empty():
                audio_queue.get()

            # Open the audio stream and start capturing.
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=16000,  # roughly 1 second of audio per block
                channels=CHANNELS,
                callback=audio_callback,
                dtype='float32'
            ):
                try:
                    while RUNNING:
                        if not audio_queue.empty():
                            audio_chunk = audio_queue.get()
                            # Transcribe using English for live display.
                            transcription = transcribe_chunk(model, processor, audio_chunk, language="en")
                            print(">>", transcription)
                            transcribed_chunks.append(transcription)
                            all_audio.extend(audio_chunk)
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    print("\nKeyboardInterrupt received. Stopping live transcription.")
                finally:
                    RUNNING = False

            # Display the full transcript (join all transcribed chunks).
            full_transcription = " ".join(transcribed_chunks)
            print("\n--- Full Transcription ---")
            print(full_transcription)

            # Prompt for translation.
            print("\nSelect a target language for translation of the full text:")
            for key, lang_code in m2m100_language_codes.items():
                print(f"{key}: {language_names.get(lang_code, lang_code)}")
            selection = input("Enter the number corresponding to your choice (or press Enter to skip translation): ").strip()

            if selection in m2m100_language_codes:
                target_lang_code = m2m100_language_codes[selection]
                print(f"\nTranslating full text to {language_names.get(target_lang_code, target_lang_code)}...")
                # Tokenize the full transcription.
                model_inputs = translation_tokenizer(full_transcription, return_tensors="pt")
                # Use get_lang_id to force the target language as the first token.
                gen_tokens = translation_model.generate(
                    **model_inputs,
                    forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang_code)
                )
                translated_text = translation_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
                print("\n--- Full Translation ---")
                print(translated_text)
            else:
                print("No valid selection made. Skipping translation mode.")

        elif user_input == 'no':
            print("Exiting program.")
            break
        else:
            print("Invalid input. Please type 'Y' to start or 'no' to stop.")

if __name__ == "__main__":
    main()
