# Sample build command:
# $ docker build --rm -t txapi:dev .

FROM python:3.9-slim as txapi-build
LABEL org.opencontainers.image.authors="Thanh Nguyen <btnguyen2k (at) gmail(dot)com>"
RUN mkdir -p /workspace
ADD requirements.txt /workspace
RUN cd /workspace && pip install -U -r requirements.txt
RUN python -c 'from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline; cache_dir="/workspace/cache"; model_name="multi-qa-mpnet-base-cos-v1"; model=AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir); tokenizer=AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)'
# RUN pip install -U -r requirements.txt \
# 	&& python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("distiluse-base-multilingual-cased-v1")' \
#     && python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("distiluse-base-multilingual-cased-v2")' \
#     && python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")' \
#     && python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")' \
#     && python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")' \
#     && python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("multi-qa-distilbert-cos-v1")' \
#     && python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("multi-qa-mpnet-base-cos-v1")'
