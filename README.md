# Speech Analyzer (Whisper + LLM)

A small Python project that transcribes an uploaded audio file with OpenAI's Whisper model (via Hugging Face Transformers) and then sends the transcript to an IBM watsonx.ai LLM (Llama 4 Maverick) to list its key points. The end-to-end flow is wrapped in a Gradio web UI.

## Features

- **Speech-to-text** with `openai/whisper-tiny.en` through the Transformers `automatic-speech-recognition` pipeline, processing audio in 30-second chunks.
- **Summarisation / key-point extraction** of the transcript with `meta-llama/llama-4-maverick-17b-128e-instruct-fp8` hosted on IBM watsonx.ai, using a LangChain prompt template.
- **Web UI** built with Gradio: upload an audio file, get the result back in a text box.
- Several standalone scripts for testing each piece on its own (transcription only, LLM only, UI only).

## Project structure

| File | Purpose |
| --- | --- |
| `speech_analyzer.py` | Full pipeline: Whisper transcription → LLM key-point summary → Gradio UI. This is the main entry point. |
| `speech2text_app.py` | Gradio UI for transcription only (no LLM step). |
| `simple_speech2text.py` | Command-line transcription of a local file named `downloaded_audio.mp3`. |
| `simple_llm.py` | Command-line check that the watsonx.ai LLM connection works. |
| `requirements.txt` | Python dependencies. |
| `.gitignore` | Ignores virtual environments, caches, audio files, Gradio `flagged/` output and `.env`. |

## How it works

```
audio file ──► Whisper (Transformers pipeline) ──► transcript text
                                                        │
                                                        ▼
                         LangChain PromptTemplate ("List the key points …")
                                                        │
                                                        ▼
                              IBM watsonx.ai  (Llama 4 Maverick)
                                                        │
                                                        ▼
                                             key points (text box)
```

## Prerequisites

- Python 3.9+ (3.10 or newer recommended)
- [FFmpeg](https://ffmpeg.org/) installed on the system so Transformers can decode `.mp3` and `.wav` files
- An IBM Cloud account with access to watsonx.ai, and an API key for it (needed only for the LLM step)
- Enough disk space and RAM/VRAM to download and run the models. Whisper tiny is small; the Llama 4 model runs remotely on watsonx.ai.

## Installation

```bash
git clone https://github.com/AlokHosamaneJP/speech-analyzer-whisper-llm.git
cd speech-analyzer-whisper-llm

python -m venv my_env
source my_env/bin/activate          # Windows: my_env\Scripts\activate

pip install -r requirements.txt
```

`my_env/` is already listed in `.gitignore`, so the virtual environment stays out of version control.

## Configuration

The LLM step connects to IBM watsonx.ai at `https://us-south.ml.cloud.ibm.com`. Before running anything that calls the LLM, set your credentials in the environment. The watsonx SDK typically reads:

```bash
export WATSONX_APIKEY="your-ibm-cloud-api-key"
```

The code currently uses `project_id="skills-network"`. This value comes from the IBM Skills Network lab environment. If you run the project with your own watsonx.ai project, replace it with your project ID in `speech_analyzer.py` and `simple_llm.py`.

Keep secrets out of the code. The `.gitignore` already excludes `.env`, so you can store credentials there if you load them with a library such as `python-dotenv`.

## Usage

### Full app (transcription + LLM summary)

```bash
python speech_analyzer.py
```

Open <http://localhost:7860>, upload an audio file, and the app returns a list of key points from the transcript.

### Transcription only (no LLM, no IBM account needed)

```bash
python speech2text_app.py
```

Then open <http://localhost:7860> and upload an audio file.

### Command-line transcription

Place an audio file named `downloaded_audio.mp3` in the project folder and run:

```bash
python simple_speech2text.py
```

The transcript is printed to the console.

### Test the LLM connection

```bash
python simple_llm.py
```

This sends a sample prompt ("How to read a book effectively?") to the model and prints the response.

## Notes and limitations

- **Model choice:** Whisper `tiny.en` is fast and light but English-only and less accurate than larger Whisper models. To change it, edit the `model=` argument in the pipeline call (for example `openai/whisper-base.en` or `openai/whisper-small`).
- **Long audio:** Audio is split into 30-second chunks (`chunk_length_s=30`) and processed in batches of 8 (`batch_size=8`).
- **Code duplication:** The Whisper pipeline setup is repeated in several files. Moving it into one shared module would make maintenance easier.
- **Model loading per request:** `speech2text_app.py` creates the Whisper pipeline on every upload. Creating it once at startup would make the UI respond faster.
- **Gradio `flagged/` folder:** Gradio may write flagged inputs to `flagged/`. It is ignored by git.
- **Network binding:** The apps bind to `0.0.0.0` on port `7860`, so they are reachable from other machines on your network. Change `server_name` to `"127.0.0.1"` for local-only access.

## Dependencies

From `requirements.txt`:

- `torch`, `transformers`: Whisper speech recognition
- `gradio`: web interface
- `langchain`, `langchain-ibm`: prompt templating and the watsonx.ai LLM wrapper
- `ibm-watsonx-ai`: IBM watsonx.ai SDK
- `requests`: HTTP utilities

## License

No license is specified in this repository. Add one if you want others to reuse the code.
