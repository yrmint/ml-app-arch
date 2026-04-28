# ML Module

This folder contains the first ML inference module for music genre classification.

## Model

The current version uses a pretrained music genre classification model:

`dima806/music_genres_classification`

This is the first demo version. The goal is to provide a working inference pipeline, not a final high-accuracy model.

## Main Function

The backend can call:

```python
predict_genre(audio_path: str) -> dict
# ML Module

This folder contains the ML inference module for music genre classification.

## Structure

```text
ml/
├── core/
│   └── config.py
├── inference/
│   └── genre_classifier.py
├── tests/
│   └── test_inference.py
└── cli.py
