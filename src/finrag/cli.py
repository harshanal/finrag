"""CLI for FinRAG."""

import argparse
from finrag.agent import answer


def main():
    parser = argparse.ArgumentParser(description="FinRAG evaluation script")
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    # TODO: load data, call answer for each example, compute and print metrics
    print("Evaluation not implemented yet")
