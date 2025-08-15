FROM python:3.11.13-slim

# 必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Poetryのインストール
ENV POETRY_VERSION=1.8.3
RUN curl -sSL https://install.python-poetry.org | python3 - \
 && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# 作業ディレクトリ
WORKDIR /workspace

# Poetry の設定（仮想環境をコンテナ内に作成）
RUN poetry config virtualenvs.in-project true
