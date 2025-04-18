"""Calculator module for FinRAG."""

from typing import List, Union

from .dsl import ArgRef, Operation


def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    return a + b


def subtract(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    return a - b


def multiply(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    return a * b


def divide(a: Union[int, float], b: Union[int, float]) -> float:
    if b == 0:
        raise ValueError("Division by zero")
    return a / b


def format_percentage(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def execute_program(ops: List[Operation]) -> Union[int, float]:
    """
    Execute a sequence of Operations and return the final numeric result.
    """
    results: List[Union[int, float]] = []
    for op in ops:
        args = []
        for arg in op.args:
            if isinstance(arg, ArgRef):
                args.append(results[arg.index])
            else:
                args.append(arg)
        func = globals().get(op.name)
        if not func:
            raise ValueError(f"Unknown operation: {op.name}")
        results.append(func(*args))
    return results[-1]
