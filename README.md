# FinRAG

FinRAG is a Conversational QA Agent prototype for multi‑step quantitative questions on financial reports (text + tables).

## Setup

```bash
pipx install poetry
git clone https://github.com/harshanal/finrag.git finrag
cd finrag
poetry install
poetry shell
```

Copy environment variables:

```bash
cp .env .env
```

## Dataset
This project uses data from the ConvFinQA dataset, which provides conversational question-answer pairs over financial documents.

Download it from: [https://github.com/yangji9181/ConvFinQA](https://github.com/yangji9181/ConvFinQA)




## Running Tests

```bash
poetry run pytest
```

## Evaluation

```bash
poetry run python eval.py --split dev --sample 5
```
