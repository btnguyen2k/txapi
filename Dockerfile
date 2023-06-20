# Sample build command:
# $ docker build --rm -t txapi:dev .

FROM python:3.9-slim as txapi-build
LABEL org.opencontainers.image.authors="Thanh Nguyen <btnguyen2k (at) gmail(dot)com>"
RUN mkdir -p /workspace
ADD requirements.txt /workspace
RUN cd /workspace && python -m venv myenv && bash -c 'source myenv/bin/activate && pip install -U -r requirements.txt'
RUN cd /workspace \
    && bash -c "source myenv/bin/activate && python -c 'from transformers import AutoTokenizer, AutoModel; cache_dir=\"/workspace/cache\"; model_name=\"sentence-transformers/multi-qa-mpnet-base-cos-v1\"; model=AutoModel.from_pretrained(model_name, cache_dir=cache_dir); tokenizer=AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)'"

FROM python:3.9-alpine as txapi-runtime
LABEL org.opencontainers.image.authors="Thanh Nguyen <btnguyen2k (at) gmail(dot)com>"
#COPY --from=txapi-build /usr/local/lib/python3.9 /usr/local/lib/python3.9
COPY --from=txapi-build /workspace /workspace
WORKDIR /workspace
