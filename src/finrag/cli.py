"""CLI for FinRAG."""

import argparse

from finrag.agent import answer


def main() -> None:
    parser = argparse.ArgumentParser(description="FinRAG evaluation script")
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
