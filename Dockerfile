FROM nvidia/cuda:11.4.2-base-ubuntu20.04

WORKDIR /code

COPY pyproject.toml pyproject.toml

RUN apt update && apt-get install -y python3.9 && apt-get install -y python3-pip

RUN pip3 install poetry --user

ENV PATH="${PATH}:/root/.local/bin"

RUN poetry install