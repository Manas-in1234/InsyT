import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import type { Analysis, Anomaly, Chart, Correlation, Finding, Metric, Profile, Trend } from '../types';
import { renderReport } from './report';

const MAX_UPLOAD = 10 * 1024 * 1024;
const MAX_ROWS = 100_000;
const MAX_COLUMNS = 100;

function round(val: number, decimals: number = 2): number {
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

function computeQuantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

function computeStats(values: number[]): { mean: number; median: number; min: number; max: number; std: number } {
  if (values.length === 0) return { mean: 0, median: 0, min: 0, max: 0, std: 0 };
  const sorted = [...values].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const sum = sorted.reduce((a, b) => a + b, 0);
  const mean = sum / sorted.length;
  const median = computeQuantile(sorted, 0.5);
  let variance = 0;
  if (sorted.length > 1) {
    variance = sorted.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / (sorted.length - 1);
  }
  const std = Math.sqrt(variance);
  return { mean, median, min, max, std };
}

function computeSkewness(values: number[], mean: number, std: number): number {
  const n = values.length;
  if (n < 3 || std === 0) return 0;
  const sumCubed = values.reduce((acc, v) => acc + Math.pow((v - mean) / std, 3), 0);
  return (n / ((n - 1) * (n - 2))) * sumCubed;
}

export async function readDatasetFile(
  file: File | Blob,
  filename: string
): Promise<{ rows: Record<string, any>[]; columns: string[]; sizeBytes: number }> {
  if (file.size > MAX_UPLOAD) {
    throw new Error('This file exceeds the 10 MB upload limit.');
  }
  if (!/\.(csv|xlsx)$/i.test(filename)) {
    throw new Error('Unsupported file type. Upload a CSV or XLSX file.');
  }

  const isCsv = filename.toLowerCase().endsWith('.csv');
  let rawRows: Record<string, any>[] = [];
  let detectedCols: string[] = [];

  if (isCsv) {
    const text = await file.text();
    if (!text.trim()) {
      throw new Error('The uploaded file is empty.');
    }
    const parsed = Papa.parse<Record<string, any>>(text, {
      header: true,
      skipEmptyLines: 'greedy',
      dynamicTyping: false,
    });
    if (parsed.errors.length > 0 && parsed.data.length === 0) {
      throw new Error('Could not read this file. Check that it is a valid CSV or XLSX spreadsheet.');
    }
    rawRows = parsed.data;
    detectedCols = parsed.meta.fields || (rawRows[0] ? Object.keys(rawRows[0]) : []);
  } else {
    const buffer = await file.arrayBuffer();
    if (buffer.byteLength === 0) {
      throw new Error('The uploaded file is empty.');
    }
    try {
      const workbook = XLSX.read(buffer, { type: 'array' });
      const firstSheetName = workbook.SheetNames[0];
      if (!firstSheetName) {
        throw new Error('The dataset has no sheets or data rows.');
      }
      const worksheet = workbook.Sheets[firstSheetName];
      rawRows = XLSX.utils.sheet_to_json<Record<string, any>>(worksheet, { defval: null });
      if (rawRows.length > 0) {
        detectedCols = Object.keys(rawRows[0]);
      }
    } catch (e) {
      throw new Error('Could not read this file. Check that it is a valid CSV or XLSX spreadsheet.');
    }
  }

  if (rawRows.length === 0 || detectedCols.length === 0) {
    throw new Error('The dataset has no data rows or columns.');
  }
  if (rawRows.length > MAX_ROWS) {
    throw new Error(`This dataset exceeds the ${MAX_ROWS.toLocaleString()}-row limit.`);
  }
  if (detectedCols.length > MAX_COLUMNS) {
    throw new Error(`This dataset exceeds the ${MAX_COLUMNS}-column limit.`);
  }

  // Deduplicate and clean column names
  const used = new Set<string>();
  const colMap: Record<string, string> = {};
  const cleanedCols: string[] = [];

  for (let i = 0; i < detectedCols.length; i++) {
    const rawName = String(detectedCols[i] || '').trim() || `Column ${i + 1}`;
    let candidate = rawName;
    let counter = 2;
    while (used.has(candidate)) {
      candidate = `${rawName} (${counter})`;
      counter++;
    }
    used.add(candidate);
    colMap[detectedCols[i]] = candidate;
    cleanedCols.push(candidate);
  }

  // Rename columns in rows
  const normalizedRows: Record<string, any>[] = rawRows.map((r) => {
    const newRow: Record<string, any> = {};
    for (const oldCol of detectedCols) {
      newRow[colMap[oldCol]] = r[oldCol];
    }
    return newRow;
  });

  return { rows: normalizedRows, columns: cleanedCols, sizeBytes: file.size };
}

export function analyzeDataset(
  rows: Record<string, any>[],
  columns: string[],
  filename: string,
  sizeBytes: number
): Analysis {
  const totalRows = rows.length;
  const totalCols = columns.length;

  // Count missing cells and duplicates
  let totalMissingCells = 0;
  const rowSignatures = new Set<string>();
  let duplicateRows = 0;

  for (const r of rows) {
    const sig = JSON.stringify(r);
    if (rowSignatures.has(sig)) {
      duplicateRows++;
    } else {
      rowSignatures.add(sig);
    }
    for (const c of columns) {
      const val = r[c];
      if (val === null || val === undefined || val === '' || (typeof val === 'number' && isNaN(val))) {
        totalMissingCells++;
      }
    }
  }

  // Infer column types
  const dateCandidates = new Set<string>();
  const numericCandidates = new Set<string>();

  for (const col of columns) {
    const lower = col.toLowerCase();
    const isDateName = ['date', 'time', 'month', 'year'].some((w) => lower.includes(w));

    let validDates = 0;
    let nonMissingCount = 0;
    let validNumbers = 0;

    for (const r of rows) {
      const v = r[col];
      if (v !== null && v !== undefined && v !== '' && !Number.isNaN(v)) {
        nonMissingCount++;
        // Check number
        const num = typeof v === 'number' ? v : Number(String(v).trim().replace(/,/g, ''));
        if (!isNaN(num) && isFinite(num) && String(v).trim() !== '') {
          validNumbers++;
        }
        // Check date if name hints or string looks like date
        if (isDateName || typeof v === 'string') {
          const timestamp = Date.parse(String(v));
          if (!isNaN(timestamp) && timestamp > 0) {
            validDates++;
          }
        }
      }
    }

    if (isDateName && validDates >= 3 && validDates >= nonMissingCount * 0.7) {
      dateCandidates.add(col);
    } else if (nonMissingCount > 0 && validNumbers >= nonMissingCount * 0.8) {
      numericCandidates.add(col);
    } else if (
      nonMissingCount === 0 &&
      ['num', 'amount', 'value', 'count', 'score', 'rate'].some((t) => lower.includes(t))
    ) {
      numericCandidates.add(col);
    }
  }

  const dateCols = Array.from(dateCandidates);
  const numericProfileCols = Array.from(numericCandidates);
  const categoricalCols = columns.filter((c) => !numericCandidates.has(c) && !dateCandidates.has(c));

  // Build Column Profiles
  const profiles: Profile[] = [];
  const colNumericValues: Record<string, number[]> = {};

  for (const col of columns) {
    const kind = dateCandidates.has(col) ? 'date' : numericCandidates.has(col) ? 'numeric' : 'categorical';
    let missingCount = 0;
    const distinctSet = new Set<any>();
    const frequencyMap: Record<string, number> = {};
    const nums: number[] = [];

    for (const r of rows) {
      const v = r[col];
      if (v === null || v === undefined || v === '' || (typeof v === 'number' && isNaN(v))) {
        missingCount++;
      } else {
        distinctSet.add(v);
        const strVal = String(v);
        frequencyMap[strVal] = (frequencyMap[strVal] || 0) + 1;
        if (kind === 'numeric') {
          const num = typeof v === 'number' ? v : Number(String(v).trim().replace(/,/g, ''));
          if (!isNaN(num) && isFinite(num)) {
            nums.push(num);
          }
        }
      }
    }

    colNumericValues[col] = nums;

    const profile: Profile = {
      name: col,
      type: kind,
      missing: missingCount,
      missing_percent: round((missingCount / totalRows) * 100, 1),
      unique: distinctSet.size,
    };

    if (kind === 'numeric') {
      profile.statistics = computeStats(nums);
    } else if (kind === 'categorical') {
      const sortedFreq = Object.entries(frequencyMap)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([value, count]) => ({ value, count }));
      profile.top_values = sortedFreq;
    }

    profiles.push(profile);
  }

  // Usable numeric columns (with > 1 distinct value)
  const numericUsable = numericProfileCols.filter((c) => {
    const p = profiles.find((x) => x.name === c);
    return p && p.unique > 1;
  });

  // KPIs
  const kpis: Metric[] = [
    { label: 'Records', value: totalRows, detail: 'rows in this dataset', format: 'count' },
    { label: 'Fields', value: totalCols, detail: 'columns analyzed', format: 'count' },
  ];

  for (const col of numericUsable.slice(0, 4)) {
    const p = profiles.find((x) => x.name === col);
    const stats = p?.statistics;
    if (p && stats && stats.mean !== null && stats.mean !== undefined && colNumericValues[col].length > 0) {
      const meanVal = stats.mean ?? 0;
      const medianVal = stats.median ?? 0;
      kpis.push({
        label: `Avg. ${col}`,
        value: round(meanVal, 2),
        detail: `median ${medianVal.toLocaleString(undefined, { maximumFractionDigits: 2 })} · ${colNumericValues[col].length.toLocaleString()} valid values`,
        format: 'number',
      });
    }
  }

  for (const col of categoricalCols) {
    const p = profiles.find((x) => x.name === col);
    if (p && p.unique > 1) {
      kpis.push({
        label: `${col} groups`,
        value: p.unique,
        detail: `distinct values in ${col}`,
        format: 'count',
      });
      if (kpis.length >= 6) break;
    }
  }

  // Charts and Trends
  const charts: Chart[] = [];
  const trends: Trend[] = [];

  // 1. Numeric distribution charts
  for (const col of numericUsable) {
    const vals = colNumericValues[col];
    if (vals.length === 0) continue;
    const sorted = [...vals].sort((a, b) => a - b);
    const minVal = sorted[0];
    const maxVal = sorted[sorted.length - 1];

    if (minVal === maxVal) {
      charts.push({
        id: `dist-${col}`,
        title: `${col} distribution`,
        type: 'bar',
        x_label: col,
        y_label: 'Records',
        data: [{ label: `${minVal}`, value: vals.length }],
      });
      continue;
    }

    const numBins = Math.min(12, Math.max(5, Math.floor(Math.sqrt(vals.length))));
    const binSize = (maxVal - minVal) / numBins;
    const counts = new Array(numBins).fill(0);

    for (const v of vals) {
      let b = Math.floor((v - minVal) / binSize);
      if (b >= numBins) b = numBins - 1;
      counts[b]++;
    }

    const chartData = counts.map((count, i) => {
      const start = minVal + i * binSize;
      const end = start + binSize;
      return {
        label: `${round(start, 1).toLocaleString()}–${round(end, 1).toLocaleString()}`,
        value: count,
      };
    });

    charts.push({
      id: `dist-${col}`,
      title: `${col} distribution`,
      type: 'bar',
      x_label: col,
      y_label: 'Records',
      data: chartData,
    });
  }

  // 2. Date-based line charts & period comparisons
  if (dateCols.length > 0) {
    const dateCol = dateCols[0];
    for (const measure of numericUsable.slice(0, 3)) {
      type DateEntry = { date: Date; value: number };
      const entries: DateEntry[] = [];

      for (const r of rows) {
        const dVal = r[dateCol];
        const mVal = r[measure];
        if (dVal && mVal !== null && mVal !== undefined && mVal !== '') {
          const parsedD = new Date(dVal);
          const num = typeof mVal === 'number' ? mVal : Number(String(mVal).trim().replace(/,/g, ''));
          if (!isNaN(parsedD.getTime()) && !isNaN(num) && isFinite(num)) {
            entries.push({ date: parsedD, value: num });
          }
        }
      }

      entries.sort((a, b) => a.date.getTime() - b.date.getTime());
      if (entries.length >= 3) {
        const uniqueDayCount = new Set(entries.map((e) => e.date.toISOString().slice(0, 10))).size;
        if (uniqueDayCount >= 2) {
          // Group by Day, Week, or Month
          const freq = uniqueDayCount > 60 ? 'day' : uniqueDayCount > 14 ? 'week' : 'month';
          const periodMap = new Map<string, { label: string; date: Date; sum: number; count: number }>();

          for (const e of entries) {
            let key = '';
            let label = '';
            if (freq === 'day') {
              key = e.date.toISOString().slice(0, 10);
              label = e.date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
            } else if (freq === 'week') {
              const d = new Date(e.date);
              const day = d.getDay();
              const diff = d.getDate() - day + (day === 0 ? -6 : 1);
              d.setDate(diff);
              key = d.toISOString().slice(0, 10);
              label = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
            } else {
              key = `${e.date.getFullYear()}-${String(e.date.getMonth() + 1).padStart(2, '0')}`;
              label = e.date.toLocaleDateString(undefined, { month: 'short', year: 'numeric' });
            }

            const current = periodMap.get(key) || { label, date: e.date, sum: 0, count: 0 };
            current.sum += e.value;
            current.count += 1;
            periodMap.set(key, current);
          }

          const periods = Array.from(periodMap.entries())
            .sort((a, b) => a[0].localeCompare(b[0]))
            .map(([key, val]) => ({
              key,
              label: val.label,
              date: val.date,
              mean: round(val.sum / val.count, 2),
              count: val.count,
            }));

          if (periods.length >= 2) {
            charts.push({
              id: `trend-${measure}`,
              title: `${measure} over time`,
              type: 'line',
              x_label: dateCol,
              y_label: `Average ${measure}`,
              data: periods.slice(-60).map((p) => ({
                label: p.label,
                value: p.mean,
                count: p.count,
              })),
            });

            for (let i = 0; i < periods.length - 1; i++) {
              const oldP = periods[i];
              const newP = periods[i + 1];
              const deltaPct = oldP.mean !== 0 ? ((newP.mean - oldP.mean) / Math.abs(oldP.mean)) * 100 : null;

              trends.push({
                column: measure,
                date_column: dateCol,
                comparison: { from_period: oldP.key, to_period: newP.key },
                from: oldP.mean,
                to: newP.mean,
                change: round(newP.mean - oldP.mean, 2),
                change_percent: deltaPct !== null ? round(deltaPct, 2) : null,
                observations: { from: oldP.count, to: newP.count },
                method: `Mean ${measure} by ${freq} period; period-over-period change`,
                description: `${measure} average changed from ${oldP.mean.toLocaleString()} to ${newP.mean.toLocaleString()} between ${oldP.label} and ${newP.label}${
                  deltaPct !== null
                    ? ` (${deltaPct >= 0 ? '+' : ''}${round(deltaPct, 1)}%).`
                    : ' (percentage change unavailable because previous value was zero).'
                }`,
              });
            }
          }
        }
      }
    }
  }

  // 3. Categorical distribution charts
  for (const col of categoricalCols.slice(0, 2)) {
    const p = profiles.find((x) => x.name === col);
    if (p && p.top_values && p.top_values.length > 1) {
      charts.push({
        id: `cat-${col}`,
        title: `Records by ${col}`,
        type: 'bar',
        x_label: col,
        y_label: 'Records',
        data: p.top_values.map((tv) => ({
          label: tv.value.slice(0, 48),
          value: tv.count,
        })),
      });
    }
  }

  // Anomalies via 1.5×IQR
  const anomalies: Anomaly[] = [];
  const findings: Finding[] = [];

  for (const col of numericUsable) {
    const vals = colNumericValues[col];
    if (vals.length < 8) continue;
    const sorted = [...vals].sort((a, b) => a - b);
    const q1 = computeQuantile(sorted, 0.25);
    const q3 = computeQuantile(sorted, 0.75);
    const iqr = q3 - q1;
    if (iqr <= 0) continue;

    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;

    let colMatches = 0;
    for (let rIdx = 0; rIdx < rows.length; rIdx++) {
      const v = rows[rIdx][col];
      if (v !== null && v !== undefined && v !== '') {
        const num = typeof v === 'number' ? v : Number(String(v).trim().replace(/,/g, ''));
        if (!isNaN(num) && (num < lowerBound || num > upperBound)) {
          colMatches++;
          if (anomalies.length < 30) {
            anomalies.push({
              column: col,
              value: num,
              row: rIdx + 2, // Account for header row (1-indexed)
              method: '1.5×IQR',
              lower_bound: round(lowerBound, 2),
              upper_bound: round(upperBound, 2),
              context: `Typical middle 50%: ${round(q1, 2).toLocaleString()}–${round(q3, 2).toLocaleString()}`,
              explanation:
                'Potential statistical outlier; investigate in context. An outlier is not automatically an error or fraud.',
            });
          }
        }
      }
    }

    if (colMatches > 0) {
      findings.push({
        title: `Unusual values in ${col}`,
        description: `${colMatches} value(s) fall outside the 1.5×IQR range.`,
        evidence: { count: colMatches, lower_bound: round(lowerBound, 2), upper_bound: round(upperBound, 2) },
        columns: [col],
        method: 'Interquartile range (1.5×IQR)',
        severity: 'review',
        strength: 'potential outlier',
        next_step: `Check the ${col} records against their source.`,
      });
    }
  }

  // Correlations
  const correlations: Correlation[] = [];
  const corrFindings: Finding[] = [];

  for (let i = 0; i < numericUsable.length; i++) {
    for (let j = i + 1; j < numericUsable.length; j++) {
      const colA = numericUsable[i];
      const colB = numericUsable[j];

      const pairA: number[] = [];
      const pairB: number[] = [];

      for (const r of rows) {
        const va = r[colA];
        const vb = r[colB];
        if (va !== null && va !== undefined && va !== '' && vb !== null && vb !== undefined && vb !== '') {
          const numA = typeof va === 'number' ? va : Number(String(va).trim().replace(/,/g, ''));
          const numB = typeof vb === 'number' ? vb : Number(String(vb).trim().replace(/,/g, ''));
          if (!isNaN(numA) && isFinite(numA) && !isNaN(numB) && isFinite(numB)) {
            pairA.push(numA);
            pairB.push(numB);
          }
        }
      }

      if (pairA.length < 8) continue;
      const setA = new Set(pairA);
      const setB = new Set(pairB);
      if (setA.size < 2 || setB.size < 2) continue;

      const meanA = pairA.reduce((a, b) => a + b, 0) / pairA.length;
      const meanB = pairB.reduce((a, b) => a + b, 0) / pairB.length;

      let num = 0;
      let denA = 0;
      let denB = 0;
      for (let k = 0; k < pairA.length; k++) {
        const da = pairA[k] - meanA;
        const db = pairB[k] - meanB;
        num += da * db;
        denA += da * da;
        denB += db * db;
      }

      if (denA === 0 || denB === 0) continue;
      const r = num / (Math.sqrt(denA) * Math.sqrt(denB));
      if (!isFinite(r)) continue;

      const strength = Math.abs(r) >= 0.7 ? 'strong' : Math.abs(r) >= 0.5 ? 'moderate' : 'weak';
      const desc = `${colA} and ${colB} move ${r > 0 ? 'together' : 'in opposite directions'} in this dataset (r=${r.toFixed(
        2
      )}, n=${pairA.length}). Correlation does not establish causation.`;

      correlations.push({
        column_a: colA,
        column_b: colB,
        coefficient: round(r, 3),
        observations: pairA.length,
        strength,
        description: desc,
      });

      if (Math.abs(r) >= 0.5) {
        corrFindings.push({
          title: `${strength.charAt(0).toUpperCase() + strength.slice(1)} association: ${colA} and ${colB}`,
          description: desc,
          evidence: { correlation: round(r, 3), observations: pairA.length },
          columns: [colA, colB],
          method: 'Pearson correlation',
          severity: 'info',
          strength,
          next_step: 'Check whether this relationship persists across relevant groups and time periods.',
        });
      }
    }
  }

  // Large period changes (MAD)
  for (const measure of numericUsable.slice(0, 3)) {
    const series = trends.filter((t) => t.column === measure && t.change_percent !== null);
    if (series.length >= 4) {
      const changes = series.map((t) => t.change_percent as number);
      const sortedChanges = [...changes].sort((a, b) => a - b);
      const medianChange = computeQuantile(sortedChanges, 0.5);
      const deviations = changes.map((c) => Math.abs(c - medianChange)).sort((a, b) => a - b);
      const mad = computeQuantile(deviations, 0.5);
      const threshold = Math.max(25.0, Math.abs(medianChange) + 3 * 1.4826 * mad);

      const unusual = series.filter((t) => Math.abs((t.change_percent as number) - medianChange) > threshold);
      if (unusual.length > 0) {
        unusual.sort(
          (a, b) => Math.abs((b.change_percent as number) - medianChange) - Math.abs((a.change_percent as number) - medianChange)
        );
        const topTrend = unusual[0];
        findings.push({
          title: `Unusually large period change in ${measure}`,
          description: `${measure} changed ${(topTrend.change_percent as number) >= 0 ? '+' : ''}${
            round(topTrend.change_percent as number, 1)
          }% between ${topTrend.comparison.from_period.slice(0, 10)} and ${topTrend.comparison.to_period.slice(
            0,
            10
          )}, compared with a median period change of ${medianChange >= 0 ? '+' : ''}${round(
            medianChange,
            1
          )}% across ${changes.length} comparisons. This flags an unusual observation, not a cause or a statistically significant effect.`,
          evidence: {
            comparison: topTrend.comparison,
            from: topTrend.from,
            to: topTrend.to,
            change_percent: topTrend.change_percent,
            median_comparison_percent: round(medianChange, 2),
            comparisons: changes.length,
          },
          columns: [topTrend.date_column, measure],
          method:
            'Period-over-period percentage change; compared with median and 3×scaled MAD, with a 25% minimum deviation threshold',
          severity: 'review',
          strength: 'unusual relative to observed periods',
          next_step: `Verify the ${measure} values for these periods and check whether the pattern repeats.`,
        });
      }
    }
  }

  findings.push(...corrFindings.slice(0, 4));

  // Category concentrations & Group contrasts
  for (const col of categoricalCols.slice(0, 2)) {
    const p = profiles.find((x) => x.name === col);
    if (p && p.top_values && p.top_values.length > 1) {
      const topCount = p.top_values[0].count;
      const nonMissing = totalRows - p.missing;
      const share = topCount / Math.max(1, nonMissing);

      if (share >= 0.5) {
        findings.push({
          title: `Records concentrated in ${p.top_values[0].value} (${col})`,
          description: `${topCount.toLocaleString()} of ${nonMissing.toLocaleString()} non-missing records (${round(
            share * 100,
            1
          )}%) belong to this category.`,
          evidence: {
            category: p.top_values[0].value,
            count: topCount,
            nonmissing_count: nonMissing,
            share_percent: round(share * 100, 1),
          },
          columns: [col],
          method: 'Top-category share of non-missing values',
          severity: 'info',
          strength: 'measured',
          next_step: `Compare outcomes by ${col} to see whether this concentration matters for the metric being studied.`,
        });
      }

      // Group contrasts with numeric fields
      for (const measure of numericUsable.slice(0, 2)) {
        const groupSums: Record<string, { sum: number; count: number }> = {};
        for (const r of rows) {
          const cat = r[col];
          const mVal = r[measure];
          if (cat !== null && cat !== undefined && cat !== '' && mVal !== null && mVal !== undefined && mVal !== '') {
            const num = typeof mVal === 'number' ? mVal : Number(String(mVal).trim().replace(/,/g, ''));
            if (!isNaN(num) && isFinite(num)) {
              const catKey = String(cat);
              const cur = groupSums[catKey] || { sum: 0, count: 0 };
              cur.sum += num;
              cur.count += 1;
              groupSums[catKey] = cur;
            }
          }
        }

        const groupList = Object.entries(groupSums)
          .map(([name, data]) => ({ name, mean: data.sum / data.count, count: data.count }))
          .sort((a, b) => a.mean - b.mean);

        if (groupList.length >= 2) {
          const low = groupList[0];
          const high = groupList[groupList.length - 1];
          const spread = high.mean - low.mean;
          const measureStats = profiles.find((x) => x.name === measure)?.statistics;
          const scale = measureStats?.std || 0;

          if (scale > 0 && Math.abs(spread) >= 0.5 * scale) {
            findings.push({
              title: `${measure} differs across ${col} groups`,
              description: `Mean ${measure} ranges from ${round(low.mean, 2).toLocaleString()} in ${low.name} to ${round(
                high.mean,
                2
              ).toLocaleString()} in ${high.name} (difference ${round(spread, 2).toLocaleString()}); group sizes range from ${Math.min(
                ...groupList.map((g) => g.count)
              )} to ${Math.max(...groupList.map((g) => g.count))}. This is descriptive and does not establish a cause.`,
              evidence: {
                lowest_group: low.name,
                lowest_mean: round(low.mean, 2),
                highest_group: high.name,
                highest_mean: round(high.mean, 2),
                difference: round(spread, 2),
                group_counts: Object.fromEntries(groupList.map((g) => [g.name, g.count])),
              },
              columns: [col, measure],
              method: `Non-missing group means; range compared with overall ${measure} standard deviation`,
              severity: 'info',
              strength: 'descriptive group contrast',
              next_step: `Review sample sizes and compare ${measure} within relevant subgroups before acting.`,
            });
          }
        }
      }
    }
  }

  // Skewness findings
  for (const col of numericUsable) {
    const vals = colNumericValues[col];
    if (vals.length < 8) continue;
    const stats = profiles.find((x) => x.name === col)?.statistics;
    if (!stats || !stats.std || stats.mean === null || stats.mean === undefined) continue;
    const meanVal = stats.mean;
    const medianVal = stats.median ?? meanVal;
    const stdVal = stats.std;
    const skew = computeSkewness(vals, meanVal, stdVal);
    if (isFinite(skew) && Math.abs(skew) >= 1) {
      findings.push({
        title: `${col} distribution is skewed`,
        description: `The sample skewness is ${skew >= 0 ? '+' : ''}${round(skew, 2)} across ${vals.length} non-missing values; the mean (${round(
          meanVal,
          2
        ).toLocaleString()}) and median (${round(medianVal, 2).toLocaleString()}) may summarize this distribution differently.`,
        evidence: {
          skewness: round(skew, 3),
          mean: round(meanVal, 2),
          median: round(medianVal, 2),
          observations: vals.length,
        },
        columns: [col],
        method: 'Sample skewness; |skewness| ≥ 1 descriptive flag',
        severity: 'info',
        strength: 'descriptive distribution shape',
        next_step: `Inspect the distribution and consider the median alongside the mean for ${col}.`,
      });
    }
  }

  // Data Quality findings
  const missingProfiles = profiles.filter((p) => p.missing > 0);
  if (missingProfiles.length > 0) {
    const largest = [...missingProfiles].sort((a, b) => b.missing_percent - a.missing_percent)[0];
    findings.push({
      title: `Missing values in ${missingProfiles.length} field(s)`,
      description: `${largest.name} has the highest missingness: ${largest.missing} of ${totalRows} rows (${largest.missing_percent}%).`,
      evidence: {
        missing_by_column: missingProfiles.map((p) => ({
          column: p.name,
          count: p.missing,
          percent: p.missing_percent,
        })),
      },
      columns: missingProfiles.map((p) => p.name),
      method: 'Null count by column',
      severity: 'review',
      strength: 'measured',
      next_step: 'Check whether missing values cluster by source, period, or category before choosing how to handle them.',
    });
  }

  if (duplicateRows > 0) {
    findings.push({
      title: 'Repeated rows detected',
      description: `${duplicateRows} row(s) exactly repeat an earlier record.`,
      evidence: {
        duplicate_rows: duplicateRows,
        percent_of_rows: round((duplicateRows / totalRows) * 100, 2),
      },
      columns: [...columns],
      method: 'Exact full-row duplicate count',
      severity: 'review',
      strength: 'measured',
      next_step: 'Confirm whether repeated records represent valid repeated events or accidental duplicates.',
    });
  }

  const constantCols = profiles.filter((p) => p.unique === 1).map((p) => p.name);
  if (constantCols.length > 0) {
    findings.push({
      title: `Constant field(s): ${constantCols.slice(0, 3).join(', ')}`,
      description: `${constantCols.length} field(s) contain only one observed non-missing value, so they cannot distinguish records in this dataset.`,
      evidence: { columns: constantCols },
      columns: constantCols,
      method: 'Distinct non-missing value count',
      severity: 'info',
      strength: 'measured',
      next_step: 'Check whether these fields are expected to vary or are useful only when combined with future data.',
    });
  }

  const highCardCols = profiles.filter(
    (p) => p.type === 'categorical' && p.unique >= Math.max(20, Math.floor(totalRows * 0.5))
  );
  if (highCardCols.length > 0) {
    findings.push({
      title: `High-cardinality category: ${highCardCols[0].name}`,
      description: `${highCardCols[0].name} has ${highCardCols[0].unique} distinct values across ${totalRows} rows; group-level summaries may be more useful than individual labels.`,
      evidence: { unique_values: highCardCols[0].unique, rows: totalRows },
      columns: [highCardCols[0].name],
      method: 'Unique-value count (at least 20 and 50% of rows)',
      severity: 'info',
      strength: 'measured',
      next_step: 'Consider grouping rare labels or extracting a meaningful shared category before comparing outcomes.',
    });
  }

  // Deduplicate and rank findings
  const severityRank: Record<string, number> = { review: 0, warning: 1, info: 2 };
  const seenSignatures = new Set<string>();
  const distinctFindings: Finding[] = [];

  for (const f of findings) {
    const sig = `${f.title}::${f.columns.join(',')}`;
    if (!seenSignatures.has(sig)) {
      seenSignatures.add(sig);
      distinctFindings.push(f);
    }
  }

  distinctFindings.sort((a, b) => {
    const rA = severityRank[a.severity] ?? 3;
    const rB = severityRank[b.severity] ?? 3;
    if (rA !== rB) return rA - rB;
    return Object.keys(b.evidence).length - Object.keys(a.evidence).length;
  });

  const rankedFindings = distinctFindings.slice(0, 12);

  // Recommendations
  const recommendations: string[] = [];
  if (totalMissingCells > 0) {
    const topMissing = missingProfiles.slice(0, 3).map((p) => `${p.name} (${p.missing})`).join(', ');
    recommendations.push(`Review missing entries in ${topMissing}.`);
  }
  if (duplicateRows > 0) {
    recommendations.push(
      `Check the ${duplicateRows.toLocaleString()} duplicate rows to confirm they are repeated records rather than valid repeated events.`
    );
  }
  for (const f of rankedFindings.slice(0, 3)) {
    if (f.next_step) {
      recommendations.push(f.next_step);
    }
  }
  if (recommendations.length === 0) {
    recommendations.push('Add time or category context if available, then compare metrics across meaningful groups.');
  }

  // Executive summary
  let executiveSummary = `This dataset contains ${totalRows.toLocaleString()} records across ${totalCols} fields, with ${totalMissingCells.toLocaleString()} missing cells and ${duplicateRows.toLocaleString()} duplicate rows. `;
  if (rankedFindings.length > 0) {
    executiveSummary += `Analysis found ${rankedFindings.length} evidence-backed observation(s), including ${anomalies.length} potential numeric outlier(s).`;
  } else {
    executiveSummary += 'No notable patterns met the analysis thresholds.';
  }

  const analysisResult: Analysis = {
    metadata: {
      filename,
      rows: totalRows,
      columns: totalCols,
      missing_cells: totalMissingCells,
      duplicate_rows: duplicateRows,
      size_bytes: sizeBytes,
      date_columns: dateCols,
      numeric_columns: numericUsable,
      categorical_columns: categoricalCols,
    },
    column_profiles: profiles,
    kpis,
    charts: charts.slice(0, 8),
    findings: rankedFindings,
    trends: trends.slice(0, 60),
    anomalies: anomalies.slice(0, 30),
    correlations: correlations.slice(0, 10),
    executive_summary: executiveSummary,
    recommendations: recommendations.slice(0, 5),
    ai_status: 'verified',
    report: '',
  };

  analysisResult.report = renderReport(analysisResult);
  return analysisResult;
}
