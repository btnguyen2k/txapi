from fastapi import FastAPI
from pydantic import BaseModel

VERSION = "0.2.0"

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

import models

api_resp_no_model = {
    "error": {
        "message": "The specified model does not exist.",
        "type": "invalid_request_error",
        "code": 404,
    }
}

api_resp_input_too_long = {
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

api_resp_embeddings = {
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

class EmbeddingsRequest(BaseModel):
    model: str
    input: str

@app.post("/embeddings", tags=["embeddings"])
async def api_embeddings(req: EmbeddingsRequest):
    model_name = req.model
    model_meta = models.hf_model_metadata(model_name)
    if model_meta == None:
        return api_resp_no_model
    if len(req.input) > model_meta["max_input_length"]:
        resp = api_resp_input_too_long.copy()
        resp["error"]["meta"] = model_meta
        resp["error"]["meta"]["model"] = model_name
        return resp

    model = models.get_model(model_name)
    tokenizer = models.get_tokenizer(model_name)
    embeddings = models.encode_embeddings(model, tokenizer, [req.input])
    resp = api_resp_embeddings.copy()
    resp["model"] = model_name
    resp["data"][0]["embedding"] = embeddings[0].tolist()
    resp["meta"] = model_meta
    return resp

class TokenCountsRequest(BaseModel):
    model: str
    input: str

api_resp_token_counts = {
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
    if model_meta == None:
        hf_model = False
        model_meta = models.openai_model_metadata(model_name)
    if model_meta == None:
        return api_resp_no_model
    if len(req.input) > model_meta["max_input_length"]:
        resp = api_resp_input_too_long.copy()
        resp["error"]["meta"] = model_meta
        resp["error"]["meta"]["model"] = model_name
        return resp

    if hf_model:
        tokenizer = models.get_tokenizer(model_name)
        encoded_input = tokenizer(req.input, padding=True)
        token_counts = len(encoded_input["input_ids"])
    else:
        token_counts = models.openai_token_counts(model_name, req.input)

    resp = api_resp_token_counts.copy()
    resp["model"] = model_name
    resp["data"][0]["token_counts"] = token_counts
    resp["meta"] = model_meta
    return resp

@app.get("/health", tags=["health"])
async def health():
    return "ok"

import uvicorn

if __name__ == "__main__":
    #uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4)
    uvicorn.run(app, host="0.0.0.0", port=8000)
