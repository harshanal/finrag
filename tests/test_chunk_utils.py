import pytest

from finrag.chunk_utils import build_candidate_chunks


def test_build_candidate_chunks_empty():
    # Empty turn yields no chunks
    assert build_candidate_chunks({}) == []


def test_build_candidate_chunks_all_fields():
    turn = {
        "pre_text": ["line1", "line2"],
        "table": [["cell1", "cell2"], ["cellA", "cellB"]],
        "post_text": ["footer1"],
    }
    chunks = build_candidate_chunks(turn)
    expected = [
        {"chunk_id": "text:pre:0", "text": "line1"},
        {"chunk_id": "text:pre:1", "text": "line2"},
        {"chunk_id": "table:row:0", "text": "cell1 | cell2"},
        {"chunk_id": "table:row:1", "text": "cellA | cellB"},
        {"chunk_id": "text:post:0", "text": "footer1"},
    ]
    assert chunks == expected
