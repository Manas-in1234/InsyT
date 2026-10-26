"""Data-driven profiling and analysis for CSV and XLSX datasets."""
from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

MAX_ROWS = 100_000
MAX_COLUMNS = 100


def _safe(value: Any) -> Any:
    """Convert NumPy/pandas values to JSON-compatible values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def read_dataset(content: bytes, filename: str) -> pd.DataFrame:
    """Read in-memory upload; bound dimensions before analysis."""
    if not content:
        raise ValueError("The uploaded file is empty.")
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix not in {"csv", "xlsx"}:
        raise ValueError("Unsupported file type. Upload a CSV or XLSX file.")
    try:
        if suffix == "csv":
            frame = pd.read_csv(BytesIO(content), encoding="utf-8-sig", on_bad_lines="error")
        else:
            frame = pd.read_excel(BytesIO(content), engine="openpyxl")
    except Exception as exc:
        raise ValueError("Could not read this file. Check that it is a valid CSV or XLSX spreadsheet.") from exc
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("The dataset has no data rows or columns.")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"This dataset exceeds the {MAX_ROWS:,}-row limit.")
    if len(frame.columns) > MAX_COLUMNS:
        raise ValueError(f"This dataset exceeds the {MAX_COLUMNS}-column limit.")
    names = [str(c).strip() or f"Column {i + 1}" for i, c in enumerate(frame.columns)]
    used: set[str] = set()
    unique_names = []
    for name in names:
        candidate, number = name, 2
        while candidate in used:
            candidate = f"{name} ({number})"
            number += 1
        unique_names.append(candidate)
        used.add(candidate)
    frame.columns = unique_names
    return frame


def _date_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)
        elif any(word in col.lower() for word in ("date", "time", "month", "year")):
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            valid = int(parsed.notna().sum())
            if valid >= 3 and valid >= int(df[col].notna().sum() * .7):
                df[col] = parsed
                candidates.append(col)
    return candidates


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    result = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            result.append(col)
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        # Treat completely empty columns as numeric only when their name hints at a measure;
        # otherwise keep an empty generic column categorical to avoid guessing its meaning.
        if converted.notna().sum() and converted.notna().sum() >= df[col].notna().sum() * .8:
            df[col] = converted
            result.append(col)
        elif df[col].isna().all() and any(token in col.lower() for token in ("num", "amount", "value", "count", "score", "rate")):
            df[col] = converted
            result.append(col)
    return result


def _profile(df: pd.DataFrame, dates: list[str], numeric: list[str]) -> list[dict[str, Any]]:
    profiles = []
    for col in df.columns:
        series = df[col]
        kind = "date" if col in dates else ("numeric" if col in numeric or pd.api.types.is_numeric_dtype(series) else "categorical")
        item: dict[str, Any] = {"name": col, "type": kind, "missing": int(series.isna().sum()),
                                "missing_percent": round(float(series.isna().mean() * 100), 1),
                                "unique": int(series.nunique(dropna=True))}
        if kind == "numeric":
            values = pd.to_numeric(series, errors="coerce").dropna()
            item["statistics"] = ({key: _safe(values.agg(key)) for key in ("mean", "median", "min", "max", "std")}
                                  if len(values) else {})
        elif kind == "categorical":
            item["top_values"] = [{"value": str(k), "count": int(v)} for k, v in series.value_counts(dropna=True).head(5).items()]
        profiles.append(item)
    return profiles


def _kpis(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> list[dict[str, Any]]:
    result = [{"label": "Records", "value": len(df), "detail": "rows in this dataset", "format": "count"},
              {"label": "Fields", "value": len(df.columns), "detail": "columns analyzed", "format": "count"}]
    for col in numeric[:4]:
        values = df[col].dropna()
        if len(values):
            result.append({"label": f"Avg. {col}", "value": round(float(values.mean()), 2),
                           "detail": f"median {values.median():,.2f} · {len(values):,} valid values", "format": "number"})
    for col in categorical:
        count = int(df[col].nunique(dropna=True))
        if count > 1:
            result.append({"label": f"{col} groups", "value": count, "detail": f"distinct values in {col}", "format": "count"})
            if len(result) >= 6:
                break
    return result


def _charts(df: pd.DataFrame, dates: list[str], numeric: list[str], categorical: list[str]) -> tuple[list, list]:
    charts, trends = [], []
    for col in numeric:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if not len(values):
            continue
        bins = min(12, max(5, int(np.sqrt(len(values))))) if values.nunique() > 1 else 1
        counts, edges = np.histogram(values, bins=bins)
        charts.append({"id": f"dist-{col}", "title": f"{col} distribution", "type": "bar", "x_label": col,
                       "y_label": "Records", "data": [{"label": f"{edges[i]:,.3g}–{edges[i+1]:,.3g}", "value": int(counts[i])} for i in range(len(counts))]})
    for date_col in dates[:1]:
        for measure in numeric[:3]:
            temp = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce"), "value": df[measure]}).dropna().sort_values("date")
            unique_dates = temp.date.dt.normalize().nunique()
            if len(temp) < 3 or unique_dates < 2:
                continue
            freq = "D" if unique_dates > 60 else "W" if unique_dates > 14 else "ME"
            grouped = temp.set_index("date").resample(freq).value.agg(["mean", "count"]).dropna()
            if len(grouped) < 2:
                continue
            charts.append({"id": f"trend-{measure}", "title": f"{measure} over time", "type": "line", "x_label": date_col,
                           "y_label": f"Average {measure}", "data": [{"label": idx.strftime("%b %d"), "value": _safe(row["mean"]), "count": int(row["count"])} for idx, row in grouped.tail(60).iterrows()]})
            for previous, current in zip(range(len(grouped) - 1), range(1, len(grouped))):
                old, new = float(grouped["mean"].iloc[previous]), float(grouped["mean"].iloc[current])
                delta_pct = ((new - old) / abs(old) * 100) if old != 0 else None
                trends.append({"column": measure, "date_column": date_col,
                               "comparison": {"from_period": grouped.index[previous].isoformat(), "to_period": grouped.index[current].isoformat()},
                               "from": _safe(old), "to": _safe(new), "change": _safe(new - old),
                               "change_percent": round(delta_pct, 2) if delta_pct is not None else None,
                               "observations": {"from": int(grouped["count"].iloc[previous]), "to": int(grouped["count"].iloc[current])},
                               "method": f"Mean {measure} by {freq} period; period-over-period change",
                               "description": (f"{measure} average changed from {old:,.3g} to {new:,.3g} between "
                                              f"{grouped.index[previous].strftime('%b %Y')} and {grouped.index[current].strftime('%b %Y')}"
                                              + (f" ({delta_pct:+.1f}%)." if delta_pct is not None else " (percentage change unavailable because the previous value was zero)."))})
    for col in categorical[:2]:
        counts = df[col].fillna("Missing").value_counts().head(10)
        if len(counts) > 1:
            charts.append({"id": f"cat-{col}", "title": f"Records by {col}", "type": "bar", "x_label": col,
                           "y_label": "Records", "data": [{"label": str(k)[:48], "value": int(v)} for k, v in counts.items()]})
    return charts[:8], trends[:60]


def _data_quality_findings(df: pd.DataFrame, profiles: list[dict[str, Any]], dates: list[str], numeric: list[str]) -> list[dict[str, Any]]:
    findings = []
    missing_fields = [p for p in profiles if p["missing"]]
    if missing_fields:
        largest = max(missing_fields, key=lambda p: p["missing_percent"])
        findings.append({"title": f"Missing values in {len(missing_fields)} field(s)",
                         "description": f"{largest['name']} has the highest missingness: {largest['missing']} of {len(df)} rows ({largest['missing_percent']}%).",
                         "evidence": {"missing_by_column": [{"column": p["name"], "count": p["missing"], "percent": p["missing_percent"]} for p in missing_fields]},
                         "columns": [p["name"] for p in missing_fields], "method": "Null count by column", "severity": "review", "strength": "measured",
                         "next_step": "Check whether missing values cluster by source, period, or category before choosing how to handle them."})
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        findings.append({"title": "Repeated rows detected", "description": f"{duplicate_count} row(s) exactly repeat an earlier record.",
                         "evidence": {"duplicate_rows": duplicate_count, "percent_of_rows": round(duplicate_count / len(df) * 100, 2)},
                         "columns": list(df.columns), "method": "Exact full-row duplicate count", "severity": "review", "strength": "measured",
                         "next_step": "Confirm whether repeated records represent valid repeated events or accidental duplicates."})
    constant = [p["name"] for p in profiles if p["unique"] == 1]
    if constant:
        findings.append({"title": f"Constant field(s): {', '.join(constant[:3])}", "description": f"{len(constant)} field(s) contain only one observed non-missing value, so they cannot distinguish records in this dataset.",
                         "evidence": {"columns": constant}, "columns": constant, "method": "Distinct non-missing value count", "severity": "info", "strength": "measured",
                         "next_step": "Check whether these fields are expected to vary or are useful only when combined with future data."})
    high_cardinality = [p for p in profiles if p["type"] == "categorical" and p["unique"] >= max(20, int(len(df) * .5))]
    if high_cardinality:
        findings.append({"title": f"High-cardinality category: {high_cardinality[0]['name']}",
                         "description": f"{high_cardinality[0]['name']} has {high_cardinality[0]['unique']} distinct values across {len(df)} rows; group-level summaries may be more useful than individual labels.",
                         "evidence": {"unique_values": high_cardinality[0]["unique"], "rows": len(df)}, "columns": [high_cardinality[0]["name"]],
                         "method": "Unique-value count (at least 20 and 50% of rows)", "severity": "info", "strength": "measured",
                         "next_step": "Consider grouping rare labels or extracting a meaningful shared category before comparing outcomes."})
    for col in df.columns:
        if col in dates or col in numeric or pd.api.types.is_numeric_dtype(df[col]) or df[col].isna().all():
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() and converted.notna().sum() >= df[col].notna().sum() * .5:
            findings.append({"title": f"Mixed numeric and text values in {col}", "description": f"{converted.notna().sum()} of {df[col].notna().sum()} non-missing values in {col} parse as numbers; the remaining values are text or other formats.",
                             "evidence": {"numeric_parse_count": int(converted.notna().sum()), "nonmissing_count": int(df[col].notna().sum())}, "columns": [col],
                             "method": "Numeric parse coverage", "severity": "review", "strength": "measured",
                             "next_step": "Inspect formats and units in this field before treating it as consistently numeric."})
    # Date-like headers that failed the explicit date inference check are surfaced as uncertainty,
    # not asserted errors.
    for col in df.columns:
        if col in dates or not any(word in col.lower() for word in ("date", "time", "month", "year")) or df[col].isna().all():
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        if parsed.notna().sum() < max(3, int(df[col].notna().sum() * .7)):
            findings.append({"title": f"Date parsing is uncertain for {col}", "description": f"Only {parsed.notna().sum()} of {df[col].notna().sum()} non-missing values parse as dates; no time trend was inferred from this field.",
                             "evidence": {"parsed_dates": int(parsed.notna().sum()), "nonmissing_count": int(df[col].notna().sum())}, "columns": [col],
                             "method": "Date parse coverage threshold (70%, minimum 3)", "severity": "review", "strength": "measured",
                             "next_step": "Standardize date formats and confirm the intended date field before using time-based analysis."})
    return findings


def _rank_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    severity_rank = {"review": 0, "warning": 1, "info": 2}
    seen: set[tuple[str, tuple[str, ...]]] = set()
    distinct = []
    for finding in findings:
        finding.setdefault("columns", [])
        finding.setdefault("strength", "descriptive")
        signature = (finding["title"], tuple(finding["columns"]))
        if signature in seen:
            continue
        seen.add(signature)
        distinct.append(finding)
    return sorted(distinct, key=lambda f: (severity_rank.get(f["severity"], 3), -len(f.get("evidence", {}))))[:12]


def analyze(df: pd.DataFrame, filename: str) -> dict[str, Any]:
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    dates = _date_columns(df)
    numeric_profile = _numeric_columns(df)
    numeric = [c for c in numeric_profile if df[c].nunique(dropna=True) > 1]
    categorical = [c for c in df.columns if c not in numeric_profile and c not in dates]
    profiles = _profile(df, dates, numeric_profile)
    kpis = _kpis(df, numeric, categorical)
    charts, trends = _charts(df, dates, numeric, categorical)

    anomalies, findings = [], []
    for col in numeric:
        values = pd.to_numeric(df[col], errors="coerce")
        valid = values.dropna()
        if len(valid) < 8:
            continue
        q1, q3 = valid.quantile([.25, .75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        mask = (values < lower) | (values > upper)
        matches = values[mask]
        for idx, value in matches.head(20).items():
            anomalies.append({"column": col, "value": _safe(value), "row": int(idx) + 2, "method": "1.5×IQR",
                              "lower_bound": _safe(lower), "upper_bound": _safe(upper),
                              "context": f"Typical middle 50%: {q1:,.3g}–{q3:,.3g}",
                              "explanation": "Potential statistical outlier; investigate in context. An outlier is not automatically an error or fraud."})
        if len(matches):
            findings.append({"title": f"Unusual values in {col}", "description": f"{len(matches)} value(s) fall outside the 1.5×IQR range.",
                             "evidence": {"count": len(matches), "lower_bound": lower, "upper_bound": upper},
                             "columns": [col], "method": "Interquartile range (1.5×IQR)", "severity": "review", "strength": "potential outlier",
                             "next_step": f"Check the {col} records against their source."})

    corr_data, corr_findings = [], []
    for index, col_a in enumerate(numeric):
        for col_b in numeric[index + 1:]:
            pair = df[[col_a, col_b]].dropna()
            if len(pair) < 8 or pair[col_a].nunique() < 2 or pair[col_b].nunique() < 2:
                continue
            coefficient = float(pair[col_a].corr(pair[col_b]))
            if not np.isfinite(coefficient):
                continue
            strength = "strong" if abs(coefficient) >= .7 else "moderate" if abs(coefficient) >= .5 else "weak"
            description = f"{col_a} and {col_b} move {'together' if coefficient > 0 else 'in opposite directions'} in this dataset (r={coefficient:.2f}, n={len(pair)}). Correlation does not establish causation."
            corr_data.append({"column_a": col_a, "column_b": col_b, "coefficient": round(coefficient, 3), "observations": len(pair), "strength": strength, "description": description})
            if abs(coefficient) >= .5:
                corr_findings.append({"title": f"{strength.capitalize()} association: {col_a} and {col_b}", "description": description,
                                     "evidence": {"correlation": round(coefficient, 3), "observations": len(pair)}, "columns": [col_a, col_b], "method": "Pearson correlation", "severity": "info", "strength": strength,
                                     "next_step": "Check whether this relationship persists across relevant groups and time periods."})
    for measure in numeric[:3]:
        series = [t for t in trends if t["column"] == measure]
        if len(series) < 4:
            continue
        changes = np.array([t["change_percent"] for t in series if t["change_percent"] is not None], dtype=float)
        if len(changes) < 4:
            continue
        median_change = float(np.median(changes))
        deviations = np.abs(changes - median_change)
        mad = float(np.median(deviations))
        threshold = max(25.0, abs(median_change) + 3 * 1.4826 * mad)
        unusual = [t for t in series if t["change_percent"] is not None and abs(t["change_percent"] - median_change) > threshold]
        if unusual:
            trend = max(unusual, key=lambda t: abs(t["change_percent"] - median_change))
            findings.append({"title": f"Unusually large period change in {measure}",
                             "description": f"{measure} changed {trend['change_percent']:+.1f}% between {trend['comparison']['from_period'][:10]} and {trend['comparison']['to_period'][:10]}, compared with a median period change of {median_change:+.1f}% across {len(changes)} comparisons. This flags an unusual observation, not a cause or a statistically significant effect.",
                             "evidence": {"comparison": trend["comparison"], "from": trend["from"], "to": trend["to"], "change_percent": trend["change_percent"], "median_comparison_percent": round(median_change, 2), "comparisons": len(changes)},
                             "columns": [trend["date_column"], measure], "method": "Period-over-period percentage change; compared with median and 3×scaled MAD, with a 25% minimum deviation threshold",
                             "severity": "review", "strength": "unusual relative to observed periods",
                             "next_step": f"Verify the {measure} values for these periods and check whether the pattern repeats."})
    findings.extend(corr_findings[:4])
    for col in categorical[:2]:
        counts = df[col].value_counts(dropna=True)
        if len(counts) > 1:
            top = int(counts.iloc[0])
            share = top / max(1, int(counts.sum()))
            if share >= .5:
                findings.append({"title": f"Records concentrated in {counts.index[0]} ({col})", "description": f"{top:,} of {int(counts.sum()):,} non-missing records ({share:.1%}) belong to this category.",
                                 "evidence": {"category": str(counts.index[0]), "count": top, "nonmissing_count": int(counts.sum()), "share_percent": round(share * 100, 1)},
                                 "columns": [col], "method": "Top-category share of non-missing values", "severity": "info", "strength": "measured",
                                 "next_step": f"Compare outcomes by {col} to see whether this concentration matters for the metric being studied."})
            value_cols = [c for c in numeric if df[c].notna().sum() >= 8]
            for measure in value_cols[:2]:
                grouped = df[[col, measure]].dropna().groupby(col)[measure].agg(["mean", "count"]).sort_values("mean")
                if len(grouped) < 2:
                    continue
                low, high = grouped.iloc[0], grouped.iloc[-1]
                spread = float(high["mean"] - low["mean"])
                scale = float(df[measure].dropna().std())
                if not np.isfinite(scale) or scale == 0 or abs(spread) < .5 * scale:
                    continue
                findings.append({"title": f"{measure} differs across {col} groups", "description": f"Mean {measure} ranges from {low['mean']:,.3g} in {grouped.index[0]} to {high['mean']:,.3g} in {grouped.index[-1]} (difference {spread:,.3g}); group sizes range from {int(grouped['count'].min())} to {int(grouped['count'].max())}. This is descriptive and does not establish a cause.",
                                 "evidence": {"lowest_group": str(grouped.index[0]), "lowest_mean": _safe(low["mean"]), "highest_group": str(grouped.index[-1]), "highest_mean": _safe(high["mean"]), "difference": spread, "group_counts": {str(k): int(v) for k, v in grouped["count"].items()}},
                                 "columns": [col, measure], "method": f"Non-missing group means; range compared with overall {measure} standard deviation",
                                 "severity": "info", "strength": "descriptive group contrast",
                                 "next_step": f"Review sample sizes and compare {measure} within relevant subgroups before acting."})

    for col in numeric:
        values = df[col].dropna()
        if len(values) < 8 or values.std() == 0:
            continue
        skew = float(values.skew())
        if np.isfinite(skew) and abs(skew) >= 1:
            findings.append({"title": f"{col} distribution is skewed", "description": f"The sample skewness is {skew:+.2f} across {len(values)} non-missing values; the mean ({values.mean():,.3g}) and median ({values.median():,.3g}) may summarize this distribution differently.",
                             "evidence": {"skewness": round(skew, 3), "mean": _safe(values.mean()), "median": _safe(values.median()), "observations": len(values)},
                             "columns": [col], "method": "Sample skewness (pandas); |skewness| ≥ 1 descriptive flag",
                             "severity": "info", "strength": "descriptive distribution shape",
                             "next_step": f"Inspect the distribution and consider the median alongside the mean for {col}."})

    findings.extend(_data_quality_findings(df, profiles, dates, numeric))
    findings = _rank_findings(findings)

    missing_by = df.isna().sum()
    recommendations = []
    if missing:
        recommendations.append("Review missing entries in " + ", ".join(f"{c} ({int(n)})" for c, n in missing_by[missing_by > 0].head(3).items()) + ".")
    if duplicates:
        recommendations.append(f"Check the {duplicates:,} duplicate rows to confirm they are repeated records rather than valid repeated events.")
    recommendations.extend(item["next_step"] for item in findings[:3])
    if not recommendations:
        recommendations = ["Add time or category context if available, then compare metrics across meaningful groups."]
    summary = f"This dataset contains {rows:,} records across {cols} fields, with {missing:,} missing cells and {duplicates:,} duplicate rows. "
    summary += f"Analysis found {len(findings)} evidence-backed observation(s), including {len(anomalies)} potential numeric outlier(s)." if findings else "No notable patterns met the analysis thresholds."
    result = {"metadata": {"filename": filename, "rows": rows, "columns": cols, "missing_cells": missing, "duplicate_rows": duplicates,
                           "size_bytes": None, "date_columns": dates, "numeric_columns": numeric, "categorical_columns": categorical},
              "column_profiles": profiles, "kpis": kpis, "charts": charts, "findings": findings, "trends": trends[:8],
              "anomalies": anomalies[:30], "correlations": corr_data[:10], "executive_summary": summary,
              "recommendations": recommendations[:5], "ai_status": "fallback"}
    return _safe(result)
