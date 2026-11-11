
import gradio as gr
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM
from transformers.utils import logging as hf_logging
import warnings

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")   
#######------------- LLM -------------####
  
# initiate LLM instance, this can be IBM WatsonX, huggingface, or OpenAI instance
  
params = {
    GenParams.MAX_NEW_TOKENS: 700,  # The maximum number of tokens that the model can generate in a single run.
    GenParams.TEMPERATURE: 0.1,    # A parameter that controls the randomness of the token generation.
}
  
llm_model = WatsonxLLM(
    model_id='meta-llama/llama-4-maverick-17b-128e-instruct-fp8',
    url="https://us-south.ml.cloud.ibm.com",
    params=params,
    project_id="skills-network",
)

#######------------- Prompt Template -------------####
temp = """List the key points with details from the following context:
{context}"""
  
pt = PromptTemplate(
    input_variables=["context"],
    template=temp
)
prompt_to_llm = pt | llm_model
  
#######------------- Speech2text -------------####
  
def transcript_audio(audio_file):
    # Initialize the speech recognition pipeline
    pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny.en",
  chunk_length_s=30,
)

  
    # Transcribe the audio file and return the result
  
    transcript_txt = pipe(audio_file, batch_size=8)["text"]
  
    # run the chain to merge transcript text with the template and send it to the LLM
    result = prompt_to_llm.invoke({"context": transcript_txt})
    return result
  
#######------------- Gradio -------------####
  
audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()
  
# Create the Gradio interface with the function, inputs, and outputs
  
iface = gr.Interface(fn=transcript_audio, 
                     inputs=audio_input, outputs=output_text, 
                     title="Audio Transcription App",
                     description="Upload the audio file")
iface.launch(server_name="0.0.0.0", server_port=7860)
