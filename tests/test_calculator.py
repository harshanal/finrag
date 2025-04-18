import pytest
from finrag.calculator import add, subtract, multiply, divide, format_percentage, execute_program
from finrag.dsl import Operation, ArgRef

def test_add_subtract_multiply_divide():
    assert add(2, 3) == 5
    assert subtract(5, 3) == 2
    assert multiply(2, 4) == 8
    assert divide(10, 2) == 5.0
    with pytest.raises(ValueError):
        divide(1, 0)

def test_format_percentage():
    assert format_percentage(0.141, decimals=1) == "14.1%"
    assert format_percentage(0.5, decimals=0) == "50%"

def test_execute_program_simple():
    ops = [
        Operation("add", [2, 3]),
        Operation("multiply", [ArgRef(0), 10]),
        Operation("divide", [ArgRef(1), 5]),
    ]
    result = execute_program(ops)
    assert result == 10
