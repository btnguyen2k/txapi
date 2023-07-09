# Sample build command:
# $ docker build --rm -t txapi:dev .

FROM python:3.9-slim as txapi-build
LABEL org.opencontainers.image.authors="Thanh Nguyen <btnguyen2k (at) gmail(dot)com>"
RUN mkdir -p /workspace
ADD requirements.txt /workspace
ADD *.py /workspace
RUN cd /workspace && python -m venv myenv && bash -c 'source myenv/bin/activate && pip install -U -r requirements.txt'
# Preload models from HuggingFace Hub
RUN cd /workspace && bash -c "source myenv/bin/activate && python _preload_hf_model.py sentence-transformers/multi-qa-mpnet-base-cos-v1"
RUN cd /workspace && bash -c "source myenv/bin/activate && python _preload_hf_model.py sentence-transformers/multi-qa-distilbert-cos-v1"
RUN cd /workspace && bash -c "source myenv/bin/activate && python _preload_hf_model.py sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
RUN cd /workspace && bash -c "source myenv/bin/activate && python _preload_hf_model.py sentence-transformers/all-mpnet-base-v2"
RUN cd /workspace && bash -c "source myenv/bin/activate && python _preload_hf_model.py sentence-transformers/all-MiniLM-L12-v2"
RUN cd /workspace && bash -c "source myenv/bin/activate && python _preload_hf_model.py sentence-transformers/all-MiniLM-L6-v2"
RUN cd /workspace && bash -c "source myenv/bin/activate && python _preload_hf_model.py vinai/bartpho-syllable"

FROM python:3.9-slim as txapi-runtime
LABEL org.opencontainers.image.authors="Thanh Nguyen <btnguyen2k (at) gmail(dot)com>"
ARG USERNAME=api
ARG USERID=1000
RUN useradd --system --create-home --home-dir /workspace --shell /bin/bash --uid $USERID $USERNAME
COPY --from=txapi-build --chown=$USERNAME /workspace /workspace
USER $USERNAME
WORKDIR /workspace
EXPOSE 8000

# Prevents Python from writing pyc files to disc (equivalent to python -B option)
ENV PYTHONDONTWRITEBYTECODE 1
# Prevents Python from buffering stdout and stderr (equivalent to python -u option)
ENV PYTHONUNBUFFERED 1
CMD ["bash", "-c", "source myenv/bin/activate && python server.py"]
