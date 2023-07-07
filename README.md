# txapi

This repository aims to package Transformers models and provide them as API to be used locally.

Latest version: [v0.3.1](RELEASE-NOTES.md).

## Features

| API                  | Description                                                   |
|----------------------|---------------------------------------------------------------|
| `POST /embeddings`   | compute and return the embeddings vector from the input text. |
| `POST /split_text`   | split long text string into smaller chunks.                   |
| `POST /token_counts` | return token counts for an input text.                        |

Included models:
- `sentence-transformers` models:
  - `all-mpnet-base-v2`
  - `all-MiniLM-L6-v2`
  - `all-MiniLM-L12-v2`
  - `multi-qa-mpnet-base-cos-v1`
  - `multi-qa-MiniLM-L6-cos-v1`
  - `multi-qa-distilbert-cos-v1`

### `POST /embeddings`

Compute and return the embeddings vector from the input text.

Params are sent as JSON via request body:

| Param   | Type (and default value) | Description                                  |
|---------|--------------------------|----------------------------------------------|
| `input` | `str`                    | the input string to extract embeddings       |
| `model` | `str`                    | name of the model used to extract embeddings |

Supported models:
- `sentence-transformers/*`

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

### `POST /split_text`

Split long text string into smaller chunks.

Params are sent as JSON via request body:

| Param           | Type (and default value) | Description                                                |
|-----------------|--------------------------|------------------------------------------------------------|
| `input`         | `str`                    | the input text to be split                                 |
| `model`         | `str`                    | name of the model used for token counting                  |
| `type`          | `str='plaintext'`        | type of the input text (*)                                 |
| `max_tokens`    | `int=0`                  | max chunk size, as number of tokens                        |
| `chunk_overlap` | `int=0`                  | specify how many tokens chunks can overlap with each other |

> (*) input text type:
> - `markdown` (or `md`): input is Markdown text; API will try its best to preserve headings.
> - `html`: input is HTML text; API will try not to break tags.
> - `html-markdown` (or `html-md`): input is HTML text; API will convert input text to Markdown before splitting.
> - other values: treat input as plain text.

Supported models:
- `sentence-transformers/*`
- [OpenAI models](https://platform.openai.com/docs/models).

```shell
curl http://localhost:8000/split_text \
  -H "Content-Type: application/json" \
  -d '{
    "input": "It is important to count the number of tokens in your prompt to optimize its performance. However, there is a caveat: token count is not directly correlated to the number of characters, bytes, or words in the prompt. This means that similar strings can result in different token counts in different natural languages (such as English and Vietnamese). As a result, accurately counting tokens is not a trivial task.\nOpenAI uses Byte Pair Encoding (BPE) for prompt input tokenization. Luckily, several programming language libraries can tokenize and count the number of tokens from a string. However, if a suitable library is unavailable in your preferred programming language, you can still estimate the number of tokens. This post proposes an approach to approximating the token count from an input string.",
    "model": "text-embedding-ada-002",
    "max_tokens": 100
  }'
```

Example response:
```json
{
    "object": "list",
    "data": [
        {
            "object": "string",
            "chunk": "It is important to count the number of tokens in your prompt to optimize its performance. However, there is a caveat: token count is not directly correlated to the number of characters, bytes, or words in the prompt. This means that similar strings can result in different token counts in different natural languages (such as English and Vietnamese). As a result, accurately counting tokens is not a trivial task.",
            "index": 0,
            "chunk_size": 79
        },
        {
            "object": "string",
            "chunk": "OpenAI uses Byte Pair Encoding (BPE) for prompt input tokenization. Luckily, several programming language libraries can tokenize and count the number of tokens from a string. However, if a suitable library is unavailable in your preferred programming language, you can still estimate the number of tokens. This post proposes an approach to approximating the token count from an input string.",
            "index": 1,
            "chunk_size": 73
        }
    ],
    "model": "text-embedding-ada-002",
    "meta": {
        "min_tokens": 16,
        "max_tokens": 8191,
        "max_input_length": 49146
    }
}
```

### `POST /token_counts`

Return token counts for an input text.

Params are sent as JSON via request body:

| Param   | Type (and default value) | Description                               |
|---------|--------------------------|-------------------------------------------|
| `input` | `str`                    | the input string to count tokens          |
| `model` | `str`                    | name of the model used for token counting |

Supported models:
- `sentence-transformers/*`
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
# API documentation is ready at http://localhost:8000/docs
```

## Datasets and Benchmark

Some datasets and benchmark results are available at [datasets](./datasets) folder.

## License

MIT - See [LICENSE.md](LICENSE.md).
