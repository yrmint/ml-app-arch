# Music genre classifier

<!-- Meta & Language -->
[![License](https://img.shields.io/github/license/yrmint/ml-app-arch)](#)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](#)

<!-- Backend Frameworks -->
[![FastAPI](https://img.shields.io/badge/FastAPI-009485?logo=fastapi&logoColor=white)](#)
[![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=Pydantic&logoColor=white)](#)

<!-- Machine Learning & Data Science -->
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?logo=huggingface&logoColor=black)](#)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](#)
[![Optuna](https://img.shields.io/badge/Optuna-002C76?logo=optuna&logoColor=white)](#)

<!-- Infrastructure, CI/CD & Tools -->
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](#)
[![Redis](https://img.shields.io/badge/Redis-DD0031?logo=redis&logoColor=white)](#)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=github-actions&logoColor=white)](#)
[![uv](https://img.shields.io/badge/uv-Lightning_Fast-261230?logo=python&logoColor=white)](#)
[![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)](#)
[![flake8](https://img.shields.io/badge/flake8-24292E?logo=python&logoColor=white)](#)

**Автоматическая классификация музыкальных жанров** с использованием алгоритмов машинного обучения

Масштабируемое веб-приложение, которое определяет вероятность принадлежности загружаемого музыкального трека к 10 музыкальным жанрам 

## About

Проект реализует классификацию музыкальных жанров. Приложение принимает аудиофайл и с помощью предобученной DL-модели прогнозирует, к какому жанру он относится. 


### Ключевые возможности

- Загрузка аудиофайлов через клиентский интерфейс.
- Асинхронная обработка аудио с использованием RabbitMQ.
- Поддержка трансформерной предобученной модели [HuggingFace: dima806/music_genres_classification](https://huggingface.co/dima806/music_genres_classification), а также легковесной кастомной CNN (по умолчанию).
- Легкое горизонтальное масштабирование через число воркеров

### Результаты

- Достигнут F1-score = 0.82 (**без утечки данных**, свойственной решениям этой бизнес-задачи)
- Вес модели сокращен с 380 МБ ([baseline transformer](https://huggingface.co/dima806/music_genres_classification)) до 2 МБ за счет перехода к кастомной PyTorch CNN модели без потери точности
- Настроен базовый пайплайн инференса, обернутый в Docker.

## Технологический Стэк

- **Язык:** Python 	
- **Пакетный менеджер:** uv
- **Контейнеризация:** Docker
- **ML / AI:** HuggingFace Transformers, PyTorch, Librosa, Scikit-learn, Optuna-dashboard
- **Backend:** FastApi, Uvicorn, Pydantic, RabbitMQ, Redis
- **Frontend:** Streamlit
- **Код-стайл и CI:** Flake8, GitHub Actions

## Установка и запуск

### Предварительные требования
- [Python](https://www.python.org/downloads/)
- Пакетный менеджер [uv](https://github.com/astral-sh/uv)
- [Docker](https://www.docker.com/)


### 1. Клонирование репозитория

```commandline
git clone https://github.com/yrmint/ml-app-arch.git
```

### 2. Установка зависимостей

```commandline
uv python install 3.11
```

```commandline
uv sync --all-groups
```

### 3. Запуск через Docker (рекомендованный способ)

```commandline
docker compose up -d --scale worker=4
```

Количество воркеров для параллельной обработки запросов можно свободно менять

### 4. Запуск клиента

```commandline
python -m frontend.start
```

## Тестирование
Linting:
```commandline
python -m flake8 .
```

Tests:
```commandline
python -m pytest .
```
