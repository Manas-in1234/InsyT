from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import MAX_UPLOAD, app
from backend.analyzer import analyze, read_dataset
from backend.report import render_report
from backend.ai_service import interpret

client = TestClient(app)


def csv_file(text: str = "value,group\n1,a\n2,b\n"):
    return {"file": ("data.csv", text.encode(), "text/csv")}


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_csv_upload_and_response_contract(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/api/analyze", files=csv_file())
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["rows"] == 2
    assert body["metadata"]["numeric_columns"] == ["value"]
    assert body["ai_status"] == "fallback"
    assert "## Limitations" in body["report"]


def test_xlsx_upload():
    stream = io.BytesIO()
    pd.DataFrame({"Amount": [2, 5], "Label": ["x", "y"]}).to_excel(stream, index=False)
    response = client.post("/api/analyze", files={"file": ("sheet.xlsx", stream.getvalue(), "application/octet-stream")})
    assert response.status_code == 200
    assert response.json()["metadata"]["rows"] == 2


@pytest.mark.parametrize("filename,content,status", [
    ("empty.csv", b"", 400),
    ("bad.txt", b"a,b\n1,2", 400),
    ("broken.csv", b'a,b\n1,"unfinished', 400),
    ("invalid.xlsx", b"not an xlsx", 400),
    ("headers.csv", b"a,b\n", 400),
])
def test_invalid_uploads(filename, content, status):
    response = client.post("/api/analyze", files={"file": (filename, content, "application/octet-stream")})
    assert response.status_code == status


def test_missing_duplicates_outlier_and_no_date():
    df = pd.DataFrame({"reading": list(range(20)) + [100, 101], "group": [f"g{i}" for i in range(20)] + [None, "unique"]})
    df = pd.concat([df, df.iloc[[20]]], ignore_index=True)
    result = analyze(df, "custom.csv")
    assert result["metadata"]["missing_cells"] == 2
    assert result["metadata"]["duplicate_rows"] == 1
    assert result["metadata"]["date_columns"] == []
    assert result["anomalies"]
    assert result["anomalies"][0]["lower_bound"] is not None
    assert result["trends"] == []


def test_constant_empty_numeric_and_column_names():
    df = pd.DataFrame({"constant": [3] * 12, "empty_num": [None] * 12, "Label": ["one", "two"] * 6})
    result = analyze(df, "varied.csv")
    assert "constant" not in result["metadata"]["numeric_columns"]
    assert "constant" not in result["metadata"]["categorical_columns"]
    assert result["anomalies"] == []
    assert "Label" in [p["name"] for p in result["column_profiles"]]
    assert result["column_profiles"][1]["statistics"] == {}


def test_correlations_include_shared_sample_count():
    df = pd.DataFrame({"alpha": range(12), "beta": [x * 2 for x in range(12)], "gamma": [None] * 6 + list(range(6))})
    result = analyze(df, "arbitrary.csv")
    pair = next(x for x in result["correlations"] if x["column_a"] == "alpha" and x["column_b"] == "beta")
    assert pair["coefficient"] == 1.0
    assert pair["observations"] == 12
    assert "causation" in pair["description"]


def test_trends_include_period_comparison_and_zero_safe():
    df = pd.DataFrame({"event_date": pd.date_range("2025-01-01", periods=12, freq="MS"), "measure": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
    result = analyze(df, "events.csv")
    assert result["trends"]
    assert result["trends"][0]["comparison"]["from_period"]
    assert result["trends"][0]["method"]
    assert result["trends"][0]["change_percent"] is None or result["trends"][0]["change_percent"] == pytest.approx(.15)


def test_dimension_and_upload_limits():
    too_many_rows = b"v\n" + b"1\n" * 100_001
    response = client.post("/api/analyze", files={"file": ("large.csv", too_many_rows, "text/csv")})
    assert response.status_code == 400
    too_many_cols = (",".join(f"c{i}" for i in range(101)) + "\n" + ",".join("1" for _ in range(101))).encode()
    response = client.post("/api/analyze", files={"file": ("wide.csv", too_many_cols, "text/csv")})
    assert response.status_code == 400
    response = client.post("/api/analyze", files={"file": ("big.csv", b"x" * (MAX_UPLOAD + 1), "text/csv")})
    assert response.status_code == 413


def test_ai_fallback_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = analyze(pd.DataFrame({"metric": range(10)}), "small.csv")
    summary, recommendations, status = interpret(result)
    assert status == "fallback"
    assert summary == result["executive_summary"]
    assert recommendations


def test_report_sections():
    result = analyze(pd.DataFrame({"measure": range(10)}), "report.csv")
    report = render_report(result)
    for section in ("Dataset profile", "Data quality", "Key statistics", "Trends", "Potential anomalies", "Correlations", "Executive summary", "Recommendations", "Limitations"):
        assert section in report


def test_synthetic_demo_csv_is_real_and_analyzable():
    sample_path = Path(__file__).resolve().parents[2] / "sample_data" / "sample_sales.csv"
    content = sample_path.read_bytes()
    response = client.post("/api/analyze", files={"file": ("insyt_sample_sales.csv", content, "text/csv")})
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["rows"] == 200
    assert body["metadata"]["duplicate_rows"] == 2
    assert body["metadata"]["missing_cells"] > 0
    assert body["charts"] and body["findings"] and body["anomalies"]


def test_small_dataset_no_anomalies_and_empty_findings():
    result = analyze(pd.DataFrame({"left": [1, 2, 3], "right": [3, 1, 2]}), "tiny.csv")
    assert result["anomalies"] == []
    assert result["trends"] == []
    assert result["correlations"] == []
    assert result["findings"] == []
    assert result["recommendations"]


def test_findings_have_explanation_evidence_method_action_and_fields():
    frame = pd.DataFrame({"segment": ["A"] * 16 + ["B"] * 4, "outcome": list(range(20))})
    result = analyze(frame, "categories.csv")
    assert result["findings"]
    for finding in result["findings"]:
        assert finding["title"] and finding["description"] and finding["evidence"]
        assert finding["method"] and finding["next_step"] and finding["severity"]
        assert "columns" in finding and "strength" in finding


def test_missing_duplicates_and_constant_quality_findings_are_deduplicated():
    frame = pd.DataFrame({"stable": ["same", "same", "same", "same"], "maybe_date": ["bad", "bad", None, "bad"], "note": ["x", "x", "y", "z"]})
    result = analyze(frame, "quality.csv")
    titles = [finding["title"] for finding in result["findings"]]
    assert any("Missing values" in title for title in titles)
    assert any("Repeated rows" in title for title in titles)
    assert any("Constant field" in title for title in titles)
    assert any("Date parsing" in title for title in titles)
    assert len(titles) == len(set(titles))
