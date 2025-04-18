import pytest
import pytest

import finrag.retriever as retriever_mod
from finrag.retriever import retrieve_evidence


# Dummy BM25 that returns scores = index
class DummyBM25:
    def __init__(self, tokenized):
        self.tokenized = tokenized

    def get_scores(self, tokens):
        return list(range(len(self.tokenized)))


# Dummy embedding store that returns constant embeddings
class DummyStore:
    def __init__(self):
        pass

    def get_embedding(self, text):
        return [1.0]

    def get_embeddings(self, texts):
        return [[1.0] for _ in texts]


# Fake response/result for rerank
class FakeResult:
    def __init__(self, index):
        self.index = index


class FakeResponse:
    def __init__(self, results):
        self.results = results


@pytest.fixture(autouse=True)
def patch_components(monkeypatch):
    # Patch BM25Okapi and EmbeddingStore
    monkeypatch.setattr(retriever_mod, "BM25Okapi", DummyBM25)
    monkeypatch.setattr(retriever_mod, "EmbeddingStore", DummyStore)

    # Patch Cohere rerank to fixed order [2,0,4]
    def fake_rerank(model, query, documents, top_n):
        return FakeResponse([FakeResult(2), FakeResult(0), FakeResult(4)])

    monkeypatch.setattr(retriever_mod.co, "rerank", fake_rerank)
    return


def test_retrieve_evidence_rerank():
    # Fake turn with 5 chunks: 2 pre_text, 2 table, 1 post_text
    turn = {"pre_text": ["a", "b"], "table": [["c1"], ["c2"]], "post_text": ["d"]}
    # Candidate IDs by index
    expected_ids = ["text:pre:0", "text:pre:1", "table:row:0", "table:row:1", "text:post:0"]
    # Retrieve top 3 (rerank stub orders [2,0,4])
    result = retrieve_evidence(turn, question="q", top_k=3, bm25_k=5)
    expected = [expected_ids[i] for i in [2, 0, 4]]
    assert result == expected
