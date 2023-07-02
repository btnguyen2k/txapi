import re
from fastapi import FastAPI
from pydantic import BaseModel

VERSION = "0.3.0"

# Initialize API server
tags_metadata = [
    {
        "name": "health",
        "description": "Endpoint for health check.",
    },
    {
        "name": "embeddings",
        "description": "Compute and return the embeddings vector from the input text. Input/Output follows OpenAI's embeddings API specifications.",
        "externalDocs": {
            "description": "OpenAI's embeddings API",
            "url": "https://platform.openai.com/docs/api-reference/embeddings",
        },
    },
    {
        "name": "split_text",
        "description": "Split input text to smaller chunks",
    },
    {
        "name": "token_counts",
        "description": "Return token counts for an input text.",
    },
]

app = FastAPI(
    openapi_tags = tags_metadata,
    title = "TxAPI",
    description = "Your own Transformers API server",
    version = VERSION,
    contact={
        "name": "Thanh Nguyen",
        "url": "https://github.com/btnguyen2k/",
        # "email": "btnguyen2k [at] gmail(dot)com",
    },
    license_info={
        "name": "MIT",
        "url": "https://github.com/btnguyen2k/txapi/blob/main/LICENSE.md",
    },
)

#----------------------------------------------------------------------#

import models

api_resp_no_model: dict = {
    "error": {
        "message": "The specified model does not exist.",
        "type": "invalid_request_error",
        "code": 404,
    }
}

api_resp_input_too_long: dict = {
    "error": {
        "message": "Input exceeds model's maximum length.",
        "type": "invalid_request_error",
        "code": 400,
        "meta": {
            "model": "<placeholder>",
            "max_input_length": -1,
            "max_tokens": -1,
        },
    }
}

#----------------------------------------------------------------------#

class EmbeddingsRequest(BaseModel):
    model: str # name of the model used to extract embeddings
    input: str # the input string to extract embeddings

api_resp_embeddings: dict = {
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [],
      "index": 0
    }
  ],
  "model": "<placeholder>",
  "usage": {
    "prompt_tokens": -1,
    "total_tokens": -1
  }
}

@app.post("/embeddings", tags=["embeddings"])
async def api_embeddings(req: EmbeddingsRequest):
    """
    api_embeddings takes an input string and return the embedding vector of the input.

    This API supports offline models only!
    """
    model_name = req.model
    model_meta = models.hf_model_metadata(model_name)
    if not bool(model_meta):
        return api_resp_no_model
    if len(req.input) > model_meta["max_input_length"]:
        resp = api_resp_input_too_long.copy()
        resp["error"]["meta"] = model_meta
        resp["error"]["meta"]["model"] = model_name
        return resp

    model = models.hf_get_model(model_name)
    tokenizer = models.hf_get_tokenizer(model_name)
    embeddings = models.encode_embeddings(model, tokenizer, [req.input])
    resp = api_resp_embeddings.copy()
    resp["model"] = model_name
    resp["data"][0]["embedding"] = embeddings[0].tolist()
    resp["meta"] = model_meta
    return resp

#----------------------------------------------------------------------#

class SplitTextRequest(BaseModel):
    model: str              # name of the model used to count tokens
    input: str              # the input text to be split
    type: str = 'plaintext' # type of the input text, default to 'plaintext'
    max_tokens: int = 0     # max chunk size, in tokens
    chunk_overlap: int = 0  # specify how many tokens chunks can overlap with each other

api_resp_split_text: dict = {
    "object": "list",
    "data": [
        {
            "object": "string",
            "chunk": "<placeholder>",
            "index": 0
        }
    ],
    "model": "<placeholder>",
}

@app.post("/split_text", tags=["split_text"])
async def api_split_text(req: SplitTextRequest):
    hf_model = True
    model_name = req.model
    model_meta = models.hf_model_metadata(model_name)
    if not bool(model_meta):
        hf_model = False
        model_meta = models.openai_model_metadata(model_name)
        if not bool(model_meta):
            return api_resp_no_model

    max_tokens = req.max_tokens
    if max_tokens <= 0 or max_tokens > model_meta['max_tokens']:
        max_tokens = model_meta['max_tokens']
    elif max_tokens < model_meta['min_tokens']:
        max_tokens = model_meta['min_tokens']
    chunk_overlap = req.chunk_overlap
    if chunk_overlap <= 0:
        chunk_overlap = 0

    if hf_model:
        tokenizer = models.hf_get_tokenizer(model_name)
        def length_func(input_text)->int:
            encoded_input = tokenizer(input_text, padding=True)
            return len(encoded_input["input_ids"])
        length_function = length_func
    else:
        def length_func(input_text)->int:
            return models.openai_token_counts(model_name, input_text)
        length_function = length_func

    req.type = req.type.lower()
    req.input = req.input.replace("\r", "\n")

    if req.type == "md" or req.type == "markdown":
        from langchain.text_splitter import MarkdownHeaderTextSplitter
        headers_to_split_on = [
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
            ("####", "H4"),
            ("#####", "H5"),
            ("######", "H6"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_chunks = markdown_splitter.split_text(req.input)
        chunks = []
        for md_chunk in md_chunks:
            text = md_chunk.page_content
            if md_chunk.metadata:
                for header in reversed(headers_to_split_on):
                    if header[1] in md_chunk.metadata:
                        text = header[0]+" "+md_chunk.metadata[header[1]] + "\n\n" + text
            chunks += models.split_text(text.strip(), length_function, max_tokens, chunk_overlap)
    else:
        chunks = models.split_text(req.input, length_function, max_tokens, chunk_overlap)

    resp = api_resp_split_text.copy()
    resp["model"] = model_name
    resp["meta"] = model_meta
    resp["data"] = []
    for i in range(len(chunks)):
        resp["data"].append({
            "object": "string",
            "chunk": chunks[i],
            "index": i,
            "chunk_size": length_func(chunks[i])
        })
    return resp

#----------------------------------------------------------------------#

class TokenCountsRequest(BaseModel):
    model: str # name of the model used for token counting
    input: str # the input string to count tokens

api_resp_token_counts: dict = {
  "object": "list",
  "data": [
    {
      "object": "number",
      "token_counts": -1,
      "index": 0
    }
  ],
  "model": "<placeholder>",
}

@app.post("/token_counts", tags=["token_counts"])
async def api_token_counts(req: TokenCountsRequest):
    hf_model = True
    model_name = req.model
    model_meta = models.hf_model_metadata(model_name)
    if not bool(model_meta):
        hf_model = False
        model_meta = models.openai_model_metadata(model_name)
        if not bool(model_meta):
            return api_resp_no_model
    if len(req.input) > model_meta["max_input_length"]:
        resp = api_resp_input_too_long.copy()
        resp["error"]["meta"] = model_meta
        resp["error"]["meta"]["model"] = model_name
        return resp

    if hf_model:
        tokenizer = models.hf_get_tokenizer(model_name)
        encoded_input = tokenizer(req.input, padding=True)
        token_counts = len(encoded_input["input_ids"])
    else:
        token_counts = models.openai_token_counts(model_name, req.input)

    resp = api_resp_token_counts.copy()
    resp["model"] = model_name
    resp["data"][0]["token_counts"] = token_counts
    resp["meta"] = model_meta
    return resp

#----------------------------------------------------------------------#

@app.get("/health", tags=["health"])
async def health():
    """
    health is the health check endpoint.
    :return:
    """
    return "ok"

#----------------------------------------------------------------------#

import uvicorn

if __name__ == "__main__":
    import os
    num_workers = int(os.getenv("NUM_WORKERS", 1))
    if num_workers < 1:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=num_workers)
