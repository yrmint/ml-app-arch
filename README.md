# Music genre classifier
**Automatic music genre classification** using machine learning models.

A web application that identifies the genre of a song based on an audio file.

## Model

The current version uses a [pretrained music genre classification model](https://huggingface.co/dima806/music_genres_classification).

## How to install
Clone repository:
```commandline
git clone https://github.com/yrmint/ml-app-arch.git
```
Install dependencies:
```commandline
uv sync --all-groups
```

## How to run
Run server as:
```commandline
uvicorn backend.app.main:app
```
Run client as:
```commandline
python -m frontend.start
```