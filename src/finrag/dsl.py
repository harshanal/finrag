"""DSL parser and serializer for FinRAG."""

import re
from typing import List, Union


class ArgRef:
    """Reference to a previous operation result by index."""
    def __init__(self, index: int) -> None:
        self.index = index

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ArgRef) and self.index == other.index

    def __repr__(self) -> str:
        return f"ArgRef({self.index})"


class Operation:
    """A single operation in a DSL program."""
    def __init__(self, name: str, args: List[Union[int, float, ArgRef]]) -> None:
        self.name = name
        self.args = args

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Operation)
            and self.name == other.name
            and self.args == other.args
        )

    def __repr__(self) -> str:
        return f"Operation({self.name!r}, {self.args})"


def parse_program(program_str: str) -> List[Operation]:
    """
    Parse a DSL program string into a list of Operation objects.

    Example:
        """subtract(206588, 181001), divide(#0, 181001)"""
    """
    ops: List[Operation] = []
    pattern = re.compile(r"(\w+)\s*\(([^)]*)\)")
    for name, args_str in pattern.findall(program_str):
        args: List[Union[int, float, ArgRef]] = []
        for arg in [a.strip() for a in args_str.split(',') if a.strip()]:
            if arg.startswith('#'):
                idx = int(arg[1:])
                args.append(ArgRef(idx))
            else:
                if '.' in arg:
                    args.append(float(arg))
                else:
                    args.append(int(arg))
        ops.append(Operation(name, args))
    return ops


def serialize_program(ops: List[Operation]) -> str:
    """
    Serialize a list of Operation objects back into a DSL program string.
    """
    parts: List[str] = []
    for op in ops:
        arg_strs: List[str] = []
        for arg in op.args:
            if isinstance(arg, ArgRef):
                arg_strs.append(f"#{arg.index}")
            else:
                arg_strs.append(str(arg))
        parts.append(f"{op.name}({', '.join(arg_strs)})")
    return ', '.join(parts)
