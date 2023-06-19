# Sample build command:
# $ docker build --rm -t txapi:dev .
FROM python:3.9-slim
LABEL org.opencontainers.image.authors="Thanh Nguyen <btnguyen2k (at) gmail(dot)com>"

ADD requirements.txt .
RUN pip install -U -r requirements.txt \
	&& python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("distiluse-base-multilingual-cased-v1")' \
    && python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("distiluse-base-multilingual-cased-v2")' \
    && python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")' \
    && python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")' \
    && python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")' \
    && python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("multi-qa-distilbert-cos-v1")' \
    && python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("multi-qa-mpnet-base-cos-v1")'
# torch torchvision transformers
