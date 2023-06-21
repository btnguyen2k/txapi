from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import tiktoken

model_alias = {
    "multi-qa-mpnet-base-cos-v1": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
    "gpt-35-turbo": "gpt-3.5-turbo", # Azure deployment name
    "gpt-35-turbo-16k": "gpt-3.5-turbo-16k", # Azure deployment name
}

#----------------------------------------------------------------------#

## OpenAI ##
openai_models_meta = {
    # chat
    "gpt-4": {"max_tokens": 8192, "max_input_length": 8192*6},
    "gpt-4-32k": {"max_tokens": 32768, "max_input_length": 32768*6},
    "gpt-3.5-turbo": {"max_tokens": 4096, "max_input_length": 4096*6},
    "gpt-3.5-turbo-16k": {"max_tokens": 16384, "max_input_length": 16384*6},
    # text
    "text-davinci-003": {"max_tokens": 4097, "max_input_length": 4097*6},
    "text-davinci-002": {"max_tokens": 4097, "max_input_length": 4097*6},
    "text-davinci-001": {"max_tokens": 4097, "max_input_length": 4097*6},
    "text-curie-001": {"max_tokens": 2049, "max_input_length": 2049*6},
    "text-babbage-001": {"max_tokens": 2049, "max_input_length": 2049*6},
    "text-ada-001": {"max_tokens": 2049, "max_input_length": 2049*6},
    "davinci": {"max_tokens": 2049, "max_input_length": 2049*6},
    "curie": {"max_tokens": 2049, "max_input_length": 2049*6},
    "babbage": {"max_tokens": 2049, "max_input_length": 2049*6},
    "ada": {"max_tokens": 2049, "max_input_length": 2049*6},
    # code
    "code-davinci-002": {"max_tokens": 8001, "max_input_length": 8001*6},
    "code-davinci-001": {"max_tokens": 8001, "max_input_length": 8001*6},
    "code-cushman-002": {"max_input_length": 8192*6},
    "code-cushman-001": {"max_input_length": 8192*6},
    "davinci-codex": {"max_input_length": 8192*6},
    "cushman-codex": {"max_input_length": 8192*6},
    # edit
    "text-davinci-edit-001": {"max_input_length": 8192*6},
    "code-davinci-edit-001": {"max_input_length": 8192*6},
    # embeddings
    "text-embedding-ada-002": {"max_input_length": 8192*6},
    # old embeddings
    "text-similarity-davinci-001": {"max_input_length": 8192*6},
    "text-similarity-curie-001": {"max_input_length": 8192*6},
    "text-similarity-babbage-001": {"max_input_length": 8192*6},
    "text-similarity-ada-001": {"max_input_length": 8192*6},
    "text-search-davinci-doc-001": {"max_input_length": 8192*6},
    "text-search-curie-doc-001": {"max_input_length": 8192*6},
    "text-search-babbage-doc-001": {"max_input_length": 8192*6},
    "text-search-ada-doc-001": {"max_input_length": 8192*6},
    "code-search-babbage-code-001": {"max_input_length": 8192*6},
    "code-search-ada-code-001": {"max_input_length": 8192*6},
    # open source
    "gpt2": {"max_input_length": 8192*6},
}

def openai_model_metadata(model_name):
    if model_name in openai_models_meta.keys():
        return openai_models_meta[model_name]
    elif model_name in model_alias.keys():
        if model_alias[model_name] in openai_models_meta.keys():
            return openai_models_meta[model_alias[model_name]]
    return None

#----------------------------------------------------------------------#

## HuggingFace ##
hf_cache_dir = "./cache_hf"
hf_models = {}
hf_tokenizers = {}
hf_models_meta = {
    "sentence-transformers/multi-qa-mpnet-base-cos-v1": {"max_tokens": 512, "max_input_length": 8192},
}

def hf_model_metadata(model_name):
    if model_name in hf_models_meta.keys():
        return hf_models_meta[model_name]
    elif model_name in model_alias.keys():
        if model_alias[model_name] in hf_models_meta.keys():
            return hf_models_meta[model_alias[model_name]]
    return None

def get_model(model_name):
    if model_name in hf_models.keys():
        return hf_models[model_name]
    elif model_name in model_alias.keys():
        if model_alias[model_name] in hf_models.keys():
            return hf_models[model_alias[model_name]]
    return None

def get_tokenizer(model_name):
    if model_name in hf_tokenizers.keys():
        return hf_tokenizers[model_name]
    elif model_name in model_alias.keys():
        if model_alias[model_name] in hf_tokenizers.keys():
            return hf_tokenizers[model_alias[model_name]]
    return None

# Mean Pooling - Take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Encode text and return embeddings vector
def encode_embeddings(model, tokenizer, texts):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)
    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

# Load models from HuggingFace Hub
for model_name in hf_models_meta:
    print("Loading model from HuggingFace Hub", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache_dir)
    hf_tokenizers[model_name] = tokenizer
    model = AutoModel.from_pretrained(model_name, cache_dir=hf_cache_dir)
    hf_models[model_name] = model
