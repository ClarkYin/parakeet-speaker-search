from eval.normalize import normalize_text

def test_lowercases_and_strips_punctuation():
    assert normalize_text("Hello, World!") == "hello world"

def test_collapses_whitespace():
    assert normalize_text("a   b\tc") == "a b c"

def test_empty_string():
    assert normalize_text("") == ""
