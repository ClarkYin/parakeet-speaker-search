import json
from eval.models import EngineScore, TranscriptResult, Word
from eval.report import ranked_table, diff_html, write_report

def test_ranked_table_sorts_by_cpwer():
    scores = [
        EngineScore("b", cpwer=0.30, wer=0.30),
        EngineScore("a", cpwer=0.10, wer=0.10),
        EngineScore("c", cpwer=None, wer=None),
    ]
    table = ranked_table(scores)
    a_idx, b_idx, c_idx = table.index("| a "), table.index("| b "), table.index("| c ")
    assert a_idx < b_idx < c_idx

def test_diff_html_marks_substitution():
    html = diff_html("the cat sat", "the dog sat")
    assert "sub" in html
    assert "dog" in html and "cat" in html

def test_write_report_emits_three_files(tmp_path):
    scores = [EngineScore("a", cpwer=0.1, wer=0.1)]
    results = [TranscriptResult("a", "hello world", [Word("hello", 0, 1), Word("world", 1, 2)])]
    write_report(str(tmp_path), scores, results, reference_text="hello world")
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.html").exists()
    data = json.loads((tmp_path / "scores.json").read_text())
    assert data[0]["engine_id"] == "a"
