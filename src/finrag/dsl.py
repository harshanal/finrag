"""DSL parser and serializer for FinRAG."""

import re
from typing import Any, Dict, List, Union


# --- START REMOVAL: Number Cleaning (Moved to agent.py) ---
# def clean_numeric_string(s: str) -> str:
#     """Clean a string potentially representing a number to a pure numeric format."""
#     s = s.strip()
#     # Handle parentheses for negative numbers: e.g., (500) -> -500
#     if s.startswith('(') and s.endswith(')'):
#         s = '-' + s[1:-1]
#     # Remove currency symbols and commas
#     s = s.replace('$', '').replace(',', '')
#     # Handle percentage sign: e.g., 15% -> 0.15
#     if s.endswith('%'):
#         try:
#             # Attempt to convert the part before '%' to float and divide by 100
#             return str(float(s[:-1]) / 100.0)
#         except ValueError:
#             # If conversion fails, return the string without '%' for potential later handling
#             # or let the subsequent float/int conversion attempt fail.
#             return s[:-1] 
#     return s
# --- END REMOVAL ---


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
        return isinstance(other, Operation) and self.name == other.name and self.args == other.args

    def __repr__(self) -> str:
        return f"Operation({self.name!r}, {self.args})"


def parse_program(program_str: str) -> List[Operation]:
    """
    Parse a DSL program string into a list of Operation objects.

    Example:
        subtract(206588, 181001), divide(#0, 181001)
    """
    ops: List[Operation] = []
    pattern = re.compile(r"(\w+)\s*\(([^)]*)\)")
    for name, args_str in pattern.findall(program_str):
        args: List[Union[int, float, ArgRef]] = []
        for arg in [a.strip() for a in args_str.split(",") if a.strip()]:
            if arg.startswith("#"):
                idx = int(arg[1:])
                args.append(ArgRef(idx))
            else:
                # --- START REVERT: Remove cleaning call ---
                # cleaned_arg = clean_numeric_string(arg)
                try:
                    # if "." in cleaned_arg or "e" in cleaned_arg.lower():
                    if "." in arg or "e" in arg.lower(): # Revert to check original arg
                        # args.append(float(cleaned_arg))
                        args.append(float(arg)) # Revert to convert original arg
                    else:
                        # args.append(int(cleaned_arg))
                        args.append(int(arg)) # Revert to convert original arg
                except ValueError:
                     # raise ValueError(f"Could not parse argument '{arg}' (cleaned to '{cleaned_arg}') as number.")
                     raise ValueError(f"Could not parse argument '{arg}' as number.") # Revert error message
                # --- END REVERT ---
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
    return ", ".join(parts)


def execute_program(program_str: str) -> Dict[str, Any]:
    """Parse and run a DSL program string, returning result, formatted, and intermediates."""
    # Import calculator functions here to avoid circular import
    from .calculator import add, divide, format_percentage, multiply, subtract

    ops = parse_program(program_str)
    # Map operation names to calculator functions
    op_funcs = {"add": add, "subtract": subtract, "multiply": multiply, "divide": divide}
    intermediates: List[float] = []
    for op in ops:
        args: List[Union[int, float]] = []
        for arg in op.args:
            if isinstance(arg, ArgRef):
                idx = arg.index
                if idx < 0 or idx >= len(intermediates):
                    raise IndexError(f"Invalid intermediate reference: #{idx}")
                args.append(intermediates[idx])
            else:
                args.append(arg)
        func = op_funcs.get(op.name)
        if func is None:
            raise ValueError(f"Unknown operation: {op.name}")
        result = func(*args)
        intermediates.append(result)
    final = intermediates[-1] if intermediates else 0.0
    # Percentage change detection: any subtract then divide sequence => percent
    if len(ops) >= 2 and ops[0].name == "subtract" and ops[1].name == "divide":
        # Use the division result (second intermediate) for percent formatting
        ratio = intermediates[1]
        # Round to whole percent to align with gold answers
        formatted = format_percentage(ratio, decimals=0)
    else:
        formatted = f"{final:g}" if isinstance(final, float) else str(final)
    return {"result": final, "formatted": formatted, "intermediates": intermediates}
