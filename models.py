from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

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

def get_model(model_name):
    if model_name in models.keys():
        return models[model_name]
    elif model_name in model_alias.keys():
        return models[model_alias[model_name]]
    else:
        return None

def get_tokenizer(model_name):
    if model_name in tokenizers.keys():
        return tokenizers[model_name]
    elif model_name in model_alias.keys():
        return tokenizers[model_alias[model_name]]
    else:
        return None

cache_dir = "./cache"
model_alias = {
    "multi-qa-mpnet-base-cos-v1": "sentence-transformers/multi-qa-mpnet-base-cos-v1"
}
models = {}
tokenizers = {}

# Load models from HuggingFace Hub
model_list = ["sentence-transformers/multi-qa-mpnet-base-cos-v1"]
for model_name in model_list:
    print("Loading model", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizers[model_name] = tokenizer
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    models[model_name] = model
