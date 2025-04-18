# FinRAG

FinRAG is a Conversational QA Agent prototype for multi‑step quantitative questions on financial reports (text + tables).

## Setup

```bash
pipx install poetry
git clone <repo> finrag
cd finrag
poetry install
poetry shell
```

Copy environment variables:

```bash
cp .env.example .env
```

## Running Tests

```bash
poetry run pytest
```

## Evaluation

```bash
poetry run python eval.py --split dev --sample 5
```
