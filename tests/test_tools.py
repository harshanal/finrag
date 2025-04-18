from finrag.tools import MathToolInput, run_math_tool


def test_run_math_tool():
    inp = MathToolInput(program="subtract(10,5), divide(#0,5)")
    result = run_math_tool(inp)
    assert result["answer"] == "100.0%"
    assert result["intermediates"] == [5, 1.0]
