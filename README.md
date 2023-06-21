# txapi

This repository aims to package Transformers models and provide them as API to be used locally.

Latest version: [v0.1.0](RELEASE-NOTES.md).

## Features

**`POST /embeddings`**: compute and return the embeddings vector from the input text.

Supported models:
- `sentence-transformers/multi-qa-mpnet-base-cos-v1` (alias `multi-qa-mpnet-base-cos-v1`)

Request/Response: follow [OpenAI embeddings API](https://platform.openai.com/docs/api-reference/embeddings) format.

Example request:
```sh
curl http://localhost:8000/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The food was delicious and the waiter...",
    "model": "multi-qa-mpnet-base-cos-v1"
  }'
```

Example response:
```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [
                0.02306254394352436,
                0.038156658411026,
                .... (768 floats total for multi-qa-mpnet-base-cos-v1)
                0.0053402939811348915
            ],
            "index": 0
        }
    ],
    "model": "multi-qa-mpnet-base-cos-v1",
    "usage": {
        "prompt_tokens": -1,
        "total_tokens": -1
    },
    "meta": {
        "max_tokens": 512,
        "max_input_length": 8192
    }
}
```

**`POST /token_counts`**: return token counts for an input text.

Supported models:
- `sentence-transformers/multi-qa-mpnet-base-cos-v1` (alias `multi-qa-mpnet-base-cos-v1`)
- [OpenAI models](https://platform.openai.com/docs/models).

Example request:
```sh
curl http://localhost:8000/token_counts \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The food was delicious and the waiter...",
    "model": "text-embedding-ada-002"
  }'
```

Example response:
```json
{
    "object": "list",
    "data": [
        {
            "object": "number",
            "token_counts": 8,
            "index": 0
        }
    ],
    "model": "text-embedding-ada-002",
    "meta": {
        "max_input_length": 49152
    }
}
```

## Usage

Spin up an instance from Docker image:
```sh
$ docker run -d -p 8000:8000 btnguyen2k/txapi
```

or, start the API from your local Python:

```sh
# (optional) create a virtual env
$ python -m venv myenv
$ source myenv/bin/activate

# fork & clone the Git repo
$ git clone ...

# install libs
$ pip install -r requirements.txt

# start the API server
$ python main.py
# or, uvicorn main:app --reload

# the API sever is ready at http://localhost:8000
#API documentation is ready at http://localhost:8000/docs
```

## License

MIT - See [LICENSE.md](LICENSE.md).
