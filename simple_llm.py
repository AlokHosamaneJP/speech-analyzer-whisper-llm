from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM

params = {
    GenParams.MAX_NEW_TOKENS: 700,  # The maximum number of tokens that the model can generate in a single run.
    GenParams.TEMPERATURE: 0.1,     # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
}

llama_model = WatsonxLLM(
    model_id='meta-llama/llama-4-maverick-17b-128e-instruct-fp8',
    url="https://us-south.ml.cloud.ibm.com",
    params=params,
    project_id="skills-network",
)

print(llama_model.invoke("How to read a book effectively?"))
