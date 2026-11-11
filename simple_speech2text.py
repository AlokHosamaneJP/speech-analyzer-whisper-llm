import warnings
import torch
from transformers import pipeline
from transformers.utils import logging as hf_logging
import warnings

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Initialize the speech-to-text pipeline from Hugging Face Transformers
# This uses the "openai/whisper-tiny.en" model for automatic speech recognition (ASR)
# The `chunk_length_s` parameter specifies the chunk length in seconds for processing
pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny.en",
  chunk_length_s=30,
)
# Define the path to the audio file that needs to be transcribed
audio_file = 'downloaded_audio.mp3'
# Perform speech recognition on the audio file
# The `batch_size=8` parameter indicates how many chunks are processed at a time
# The result is stored in `prediction` with the key "text" containing the transcribed text
result = pipe(audio_file, batch_size=8)["text"]
# Print the transcribed text to the console
print(result)
