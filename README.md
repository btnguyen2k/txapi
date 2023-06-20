# txapi

This repository aims to package Transformers models and provide them as API to be used locally.

Latest version: [v0.1.0](RELEASE-NOTES.md).

## Features
- Packaged models:
  - `sentence-transformers/multi-qa-mpnet-base-cos-v1`
- API:
  - `/embeddings`: following [OpenAI embeddings API](https://platform.openai.com/docs/api-reference/embeddings) format.

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
```

## License

MIT - See [LICENSE.md](LICENSE.md).
