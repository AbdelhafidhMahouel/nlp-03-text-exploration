"""Offline smoke test for the manufacturing CI corpus analysis.

Author: Abdelhafidh Mahouel
Date: 2026-11
"""

import runpy

import matplotlib.pyplot as plt


def test_ci_corpus_has_token_and_context_features(monkeypatch, tmp_path):
    """Execute the full custom analysis against the CSV corpus."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))

    result = runpy.run_module("nlp.nlp_corpus_explore_abdel", run_name="__main__")

    # Corpus: loaded from CSV, four categories, at least three documents each.
    categories = {"quality", "maintenance", "safety", "lean"}
    assert set(result["top_per_category_dict"]) == categories
    for category in categories:
        count = sum(1 for d in result["corpus"] if d["category"] == category)
        assert count >= 3

    # Cleaning: tokenize keeps words, prepare also removes stop words.
    assert result["tokenize"]("The LINE, stopped!") == ["the", "line", "stopped"]
    assert result["prepare"]("The LINE, stopped!") == ["line", "stopped"]

    # Bigrams and co-occurrence reflect the new corpus.
    assert ("preventive", "maintenance") in result["bigrams_list"]
    assert "downtime" in result["co_occurrence_dict"]["line"]

    # A larger window never finds fewer context words.
    by_window = result["co_occurrence_by_window"]
    for target in result["TARGET_TOKENS"]:
        assert len(by_window[3][target]) >= len(by_window[2][target])

    # The chart is saved for the documentation report.
    chart = tmp_path / "docs" / "images" / "ci_top_tokens_by_category_abdel.png"
    assert chart.exists()
