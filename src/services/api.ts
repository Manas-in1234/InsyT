import axios from 'axios';
import type { Analysis, MLResult } from '../types';
import { analyzeDataset, readDatasetFile } from './analyzer';
import { trainRegressor } from './mlService';

const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

// Cache parsed dataset for fast predictive modeling
let cachedDataset: {
  filename: string;
  rows: Record<string, any>[];
  columns: string[];
  sizeBytes: number;
} | null = null;

export async function analyzeUploadedFile(file: File): Promise<Analysis> {
  // If remote API is configured, attempt backend first
  if (API_BASE) {
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await axios.post<Analysis>(`${API_BASE}/api/analyze`, fd, { timeout: 60000 });
      return res.data;
    } catch (e) {
      console.warn('Remote backend call failed, falling back to local analysis engine', e);
    }
  }

  // Local in-browser analytics engine
  const { rows, columns, sizeBytes } = await readDatasetFile(file, file.name);
  cachedDataset = {
    filename: file.name,
    rows,
    columns,
    sizeBytes,
  };
  return analyzeDataset(rows, columns, file.name, sizeBytes);
}

export async function trainModel(
  file: File,
  targetColumn: string,
  featureColumns: string[],
  testSize: number = 0.2,
  nEstimators: number = 100
): Promise<MLResult> {
  // If remote API is configured, attempt backend first
  if (API_BASE) {
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('target_column', targetColumn);
      form.append('feature_columns', JSON.stringify(featureColumns));
      form.append('test_size', String(testSize));
      form.append('n_estimators', String(nEstimators));
      const res = await axios.post<MLResult>(`${API_BASE}/api/ml/train`, form, { timeout: 60000 });
      return res.data;
    } catch (e) {
      console.warn('Remote backend call failed, falling back to local ML engine', e);
    }
  }

  // Local in-browser ML engine
  let rows: Record<string, any>[];
  let columns: string[];
  let sizeBytes: number;

  if (cachedDataset && cachedDataset.filename === file.name) {
    rows = cachedDataset.rows;
    columns = cachedDataset.columns;
    sizeBytes = cachedDataset.sizeBytes;
  } else {
    const parsed = await readDatasetFile(file, file.name);
    rows = parsed.rows;
    columns = parsed.columns;
    sizeBytes = parsed.sizeBytes;
    cachedDataset = {
      filename: file.name,
      rows,
      columns,
      sizeBytes,
    };
  }

  return trainRegressor(rows, columns, targetColumn, featureColumns, testSize, nEstimators, file.name, sizeBytes);
}
