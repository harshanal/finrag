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

## Environment Variables

This project requires API keys from OpenAI, Pinecone, and Cohere.

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Fill in the API keys in the `.env` file.

    ```bash
    OPENAI_API_KEY=your-openai-key
    PINECONE_API_KEY=your-pinecone-key
    COHERE_API_KEY=your-cohere-key
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
