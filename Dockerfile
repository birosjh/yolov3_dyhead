FROM nvidia/cuda:11.4.2-base-ubuntu20.04

WORKDIR /code

COPY pyproject.toml pyproject.toml

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -yq ffmpeg libsm6 libxext6 python3.9 python3-pip python3-dev wget tmux

RUN pip3 install poetry --user

ENV PATH="${PATH}:/root/.local/bin"

RUN poetry install

RUN pip install opencv-python