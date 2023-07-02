from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import tiktoken
import re

model_alias = {
    "multi-qa-mpnet-base-cos-v1": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
    "gpt-35-turbo": "gpt-3.5-turbo", # Azure deployment name
    "gpt-35-turbo-16k": "gpt-3.5-turbo-16k", # Azure deployment name
}

#----------------------------------------------------------------------#

## OpenAI ##
openai_models_meta = {
    # chat
    "gpt-4": {"min_tokens": 16, "max_tokens": 8192, "max_input_length": 8192*6},
    "gpt-4-32k": {"min_tokens": 16, "max_tokens": 32768, "max_input_length": 32768*6},
    "gpt-3.5-turbo": {"min_tokens": 16, "max_tokens": 4096, "max_input_length": 4096*6},
    "gpt-3.5-turbo-16k": {"min_tokens": 16, "max_tokens": 16384, "max_input_length": 16384*6},
    # text
    "text-davinci-003": {"min_tokens": 16, "max_tokens": 4097, "max_input_length": 4097*6},
    "text-davinci-002": {"min_tokens": 16, "max_tokens": 4097, "max_input_length": 4097*6},
    "text-davinci-001": {"min_tokens": 16, "max_tokens": 4097, "max_input_length": 4097*6},
    "text-curie-001": {"min_tokens": 16, "max_tokens": 2049, "max_input_length": 2049*6},
    "text-babbage-001": {"min_tokens": 16, "max_tokens": 2049, "max_input_length": 2049*6},
    "text-ada-001": {"min_tokens": 16, "max_tokens": 2049, "max_input_length": 2049*6},
    "davinci": {"min_tokens": 16, "max_tokens": 2049, "max_input_length": 2049*6},
    "curie": {"min_tokens": 16, "max_tokens": 2049, "max_input_length": 2049*6},
    "babbage": {"min_tokens": 16, "max_tokens": 2049, "max_input_length": 2049*6},
    "ada": {"min_tokens": 16, "max_tokens": 2049, "max_input_length": 2049*6},
    # code
    "code-davinci-002": {"min_tokens": 16, "max_tokens": 8001, "max_input_length": 8001*6},
    "code-davinci-001": {"min_tokens": 16, "max_tokens": 8001, "max_input_length": 8001*6},
    "code-cushman-002": {"min_tokens": 16, "max_tokens": 2048, "max_input_length": 2048*6},
    "code-cushman-001": {"min_tokens": 16, "max_tokens": 2048, "max_input_length": 2048*6},
    "davinci-codex": {"min_tokens": 16, "max_input_length": 8192*6},
    "cushman-codex": {"min_tokens": 16, "max_input_length": 8192*6},
    # edit
    "text-davinci-edit-001": {"min_tokens": 16, "max_input_length": 8192*6},
    "code-davinci-edit-001": {"min_tokens": 16, "max_input_length": 8192*6},
    # embeddings
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/models#embeddings-models-1
    "text-embedding-ada-002": {"min_tokens": 16, "max_tokens": 8191, "max_input_length": 8191*6},
    # old embeddings
    "text-similarity-davinci-001": {"min_tokens": 16, "max_tokens": 2046, "max_input_length": 2046*6},
    "text-similarity-curie-001": {"min_tokens": 16, "max_tokens": 2046, "max_input_length": 2046*6},
    "text-similarity-babbage-001": {"min_tokens": 16, "max_tokens": 2046, "max_input_length": 2046*6},
    "text-similarity-ada-001": {"min_tokens": 16, "max_tokens": 2046, "max_input_length": 2046*6},
    "text-search-davinci-doc-001": {"min_tokens": 16, "max_tokens": 2046, "max_input_length": 2046*6},
    "text-search-curie-doc-001": {"min_tokens": 16, "max_tokens": 2046, "max_input_length": 2046*6},
    "text-search-babbage-doc-001": {"min_tokens": 16, "max_tokens": 2046, "max_input_length": 2046*6},
    "text-search-ada-doc-001": {"min_tokens": 16, "max_tokens": 2046, "max_input_length": 2046*6},
    "code-search-babbage-code-001": {"min_tokens": 16, "max_tokens": 2046, "max_input_length": 2046*6},
    "code-search-ada-code-001": {"min_tokens": 16, "max_tokens": 2046, "max_input_length": 2046*6},
    # open source
    # "gpt2": {"max_input_length": 8192*6},
}

def openai_model_metadata(model_name: str) -> dict:
    """
    openai_model_metadata returns the model metadata for an OpenAI model.
    :param model_name: name of the model to fetch metadata
    :return: the model's metadata, or {} if the model does not exist
    """
    if model_name in openai_models_meta.keys():
        return openai_models_meta[model_name]
    elif model_name in model_alias.keys():
        if model_alias[model_name] in openai_models_meta.keys():
            return openai_models_meta[model_alias[model_name]]
    return {}

def openai_token_counts(model_name: str, input: str) -> int:
    """
    openai_token_counts uses an OpenAI model to calculate token counts from the input string.
    :param model_name: name of the model used to calculate token counts
    :param input: the input string to count tokens
    :return: the token counts as an integer
    """
    enc = tiktoken.encoding_for_model(model_name)
    ids = enc.encode(input)
    return len(ids)

#----------------------------------------------------------------------#

## HuggingFace ##
hf_cache_dir = "./cache_hf"
hf_models = {}
hf_tokenizers = {}
hf_models_meta = {
    "sentence-transformers/multi-qa-mpnet-base-cos-v1": {"min_tokens": 16, "max_tokens": 512, "max_input_length": 8192},
}

def hf_model_metadata(model_name) -> dict:
    """
    hf_model_metadata returns the model metadata for an HuggingFace model.
    :param model_name: name of the model to fetch metadata
    :return: the model's metadata, or {} if the model does not exist
    """
    if model_name in hf_models_meta.keys():
        return hf_models_meta[model_name]
    elif model_name in model_alias.keys():
        if model_alias[model_name] in hf_models_meta.keys():
            return hf_models_meta[model_alias[model_name]]
    return {}

def hf_get_model(model_name):
    """
    hf_get_model returns a HuggingFace model.
    :param model_name: name of the model to fetch metadata
    :return: the model, or None if the model does not exist
    """
    if model_name in hf_models.keys():
        return hf_models[model_name]
    elif model_name in model_alias.keys():
        if model_alias[model_name] in hf_models.keys():
            return hf_models[model_alias[model_name]]
    return None

def hf_get_tokenizer(model_name):
    """
    hf_get_model returns a HuggingFace tokenizer.
    :param model_name: name of the model to fetch metadata
    :return: the tokenizer, or None if the model does not exist
    """
    if model_name in hf_tokenizers.keys():
        return hf_tokenizers[model_name]
    elif model_name in model_alias.keys():
        if model_alias[model_name] in hf_tokenizers.keys():
            return hf_tokenizers[model_alias[model_name]]
    return None

def mean_pooling(model_output, attention_mask):
    """
    mean_pooling calculates average of all tokens.
    :param model_output:
    :param attention_mask:
    :return:
    """
    token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_embeddings(model, tokenizer, texts):
    """
    encode_embeddings encodes the input text and returns the embedding vector.
    :param model: the model used to extract embedding vector
    :param tokenizer: the tokenizer used to tokenize input string
    :param texts: the input string, or array of string, to extract embedding vector(s)
    :return: the extracted embedding vector(s)
    """
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

#----------------------------------------------------------------------#

def split_text(text: str, length_function, chunk_size: int, chunk_overlap: int = 0, language: str = '') -> [str]:
    """
    split_text is utility function to split a long text string into smaller chunks.
    :param text: the input text to be split
    :param length_function: function to calculate text length, the function must take in a string and return an int
    :param chunk_size: maximum chunk size (chunk size is calculated using length_function)
    :param chunk_overlap: specify how much chunks can overlap with each other
    :param language: if specified, treat input as source code instead of plain text
    :return:
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
    if language == '':
        text_splitter = RecursiveCharacterTextSplitter(
            separators = ["\n\n+", "\n", "[\\.!\\?]\s+", "[,:;]\s+", "\s+", ""],
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = length_function,
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter.from_language(Language(language),
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = length_function,
        )
    chunks = text_splitter.split_text(text.replace("\r", "\n"))
    for i in range(len(chunks)):
        if i > 0 and re.match("^[.!?,:;]", chunks[i]):
            chunks[i-1] += chunks[i][0]
            chunks[i] = chunks[i][1:]
        chunks[i] = chunks[i].strip()
    return chunks

# Load models from HuggingFace Hub
for model_name in hf_models_meta:
    print("Loading model from HuggingFace Hub", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache_dir)
    hf_tokenizers[model_name] = tokenizer
    model = AutoModel.from_pretrained(model_name, cache_dir=hf_cache_dir)
    hf_models[model_name] = model
