from fastapi import FastAPI
from pydantic import BaseModel

class EmbeddingsRequest(BaseModel):
    model: str
    input: str

VERSION = "0.1.0"

# Initialize API server
tags_metadata = [
    {
        "name": "health",
        "description": "Endpoint for health check.",
    },
    {
        "name": "embeddings",
        "description": "Compute and return the embeddings vector from the input. Input/Output follows OpenAI's embeddings API specifications.",
        "externalDocs": {
            "description": "OpenAI's embeddings API",
            "url": "https://platform.openai.com/docs/api-reference/embeddings",
        },
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
        "type": "invalid_request_error"
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

@app.post("/embeddings", tags=["embeddings"])
async def api_embeddings(req: EmbeddingsRequest):
    model_name = req.model
    model = models.get_model(model_name)
    tokenizer = models.get_tokenizer(model_name)
    if model == None or tokenizer == None:
        return api_resp_no_model
    
    embeddings = models.encode_embeddings(model, tokenizer, [req.input])
    resp = api_resp_embeddings.copy()
    resp["model"] = model_name
    resp["data"][0]["embedding"] = embeddings[0]
    return resp

@app.get("/health", tags=["health"])
async def health():
    return "ok"

import uvicorn

if __name__ == "__main__":
    #uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4)
    uvicorn.run(app, host="0.0.0.0", port=8000)
