# Sample build command:
# $ docker build --rm -t txapi:dev .
FROM python:3.9-slim
LABEL org.opencontainers.image.authors="Thanh Nguyen <btnguyen2k (at) gmail(dot)com>"

ADD requirements.txt .
RUN pip install -U -r requirements.txt \
	&& python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("distiluse-base-multilingual-cased-v1"); SentenceTransformer("distiluse-base-multilingual-cased-v2")'
# torch torchvision transformers
