"""计算器：必须安全（不调 eval）、能处理日常四则与括号。"""

from ragdoc.agent.tools import safe_calculate


def test_basic_arithmetic():
    assert safe_calculate("1 + 1") == "2"
    assert safe_calculate("2 * 3.5") == "7"
    assert safe_calculate("(600 - 400) * 2") == "400"
    assert safe_calculate("10 ** 3") == "1000"


def test_float_trimmed():
    # 7.0 -> 7，2.5 保留小数
    assert safe_calculate("2.5 * 2") in {"5", "5.0", "5.00000"}


def test_rejects_code_injection():
    # 试图把语句塞进去：必须报错
    out = safe_calculate("__import__('os').system('echo pwned')")
    assert "失败" in out or "不支持" in out


def test_rejects_function_calls():
    out = safe_calculate("open('x')")
    assert "失败" in out or "不支持" in out


def test_invalid_syntax_returns_error_string():
    out = safe_calculate("1 +")
    assert "失败" in out
