import pytest

from finrag.dsl import (
    ArgRef,
    Operation,
    execute_program,
    parse_program,
    serialize_program,
)


def test_parse_and_serialize() -> None:
    program = "add(1,2),multiply(#0,3)"
    ops = parse_program(program)
    assert ops == [Operation("add", [1, 2]), Operation("multiply", [ArgRef(0), 3])]
    assert serialize_program(ops) == "add(1, 2), multiply(#0, 3)"


def test_execute_program() -> None:
    program = "subtract(206588, 181001), divide(#0, 181001)"
    res = execute_program(program)
    # Check intermediates and formatted output
    assert res["intermediates"][0] == 206588 - 181001
    assert isinstance(res["intermediates"][1], float)
    assert res["formatted"] == "14.1%"


def test_division_by_zero() -> None:
    with pytest.raises(ValueError):
        execute_program("divide(1, 0)")


def test_invalid_ref() -> None:
    with pytest.raises(IndexError):
        execute_program("add(1,2), multiply(#99,3)")
