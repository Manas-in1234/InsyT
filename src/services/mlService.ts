import type { MLResult } from '../types';
import { analyzeDataset } from './analyzer';
import { renderMLSection, renderReport } from './report';

function round(val: number, decimals: number = 6): number {
  const f = Math.pow(10, decimals);
  return Math.round(val * f) / f;
}

// Seeded pseudo-random number generator (Mulberry32) for deterministic splits and trees
function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type TreeNode =
  | { isLeaf: true; value: number }
  | {
      isLeaf: false;
      featureIdx: number;
      threshold: number;
      left: TreeNode;
      right: TreeNode;
    };

function buildDecisionTree(
  X: number[][],
  y: number[],
  indices: number[],
  depth: number,
  maxDepth: number,
  minSamplesLeaf: number,
  featureImportances: Float64Array,
  featureSubsetSize: number,
  rng: () => number
): TreeNode {
  const n = indices.length;
  if (n === 0) return { isLeaf: true, value: 0 };

  let sum = 0;
  for (let i = 0; i < n; i++) sum += y[indices[i]];
  const mean = sum / n;

  if (depth >= maxDepth || n <= minSamplesLeaf * 2) {
    return { isLeaf: true, value: mean };
  }

  // Calculate current node variance / SSE
  let currentSSE = 0;
  for (let i = 0; i < n; i++) {
    const diff = y[indices[i]] - mean;
    currentSSE += diff * diff;
  }

  if (currentSSE <= 1e-10) {
    return { isLeaf: true, value: mean };
  }

  const numFeatures = X[0].length;
  // Choose random subset of features
  const allFeatureIndices = Array.from({ length: numFeatures }, (_, i) => i);
  // Shuffle subset
  for (let i = allFeatureIndices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const temp = allFeatureIndices[i];
    allFeatureIndices[i] = allFeatureIndices[j];
    allFeatureIndices[j] = temp;
  }
  const chosenFeatures = allFeatureIndices.slice(0, featureSubsetSize);

  let bestGain = 0;
  let bestFeature = -1;
  let bestThreshold = 0;
  let bestLeftIndices: number[] = [];
  let bestRightIndices: number[] = [];

  for (const fIdx of chosenFeatures) {
    // Collect feature values and sort
    const featVals = indices.map((idx) => ({ feat: X[idx][fIdx], target: y[idx], originalIdx: idx }));
    featVals.sort((a, b) => a.feat - b.feat);

    let leftSum = 0;
    let leftCount = 0;
    let rightSum = sum;
    let rightCount = n;

    for (let i = 0; i < featVals.length - 1; i++) {
      const cur = featVals[i];
      const next = featVals[i + 1];

      leftSum += cur.target;
      leftCount++;
      rightSum -= cur.target;
      rightCount--;

      if (cur.feat === next.feat) continue;
      if (leftCount < minSamplesLeaf || rightCount < minSamplesLeaf) continue;

      const leftMean = leftSum / leftCount;
      const rightMean = rightSum / rightCount;

      // Variance reduction = currentSSE - (leftSSE + rightSSE)
      // SSE = sum(y^2) - N * mean^2, so reduction is (leftCount * leftMean^2 + rightCount * rightMean^2 - n * mean^2)
      const gain = leftCount * leftMean * leftMean + rightCount * rightMean * rightMean - n * mean * mean;

      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = fIdx;
        bestThreshold = (cur.feat + next.feat) / 2;
      }
    }
  }

  if (bestGain <= 0 || bestFeature === -1) {
    return { isLeaf: true, value: mean };
  }

  // Partition indices
  for (let i = 0; i < n; i++) {
    const idx = indices[i];
    if (X[idx][bestFeature] <= bestThreshold) {
      bestLeftIndices.push(idx);
    } else {
      bestRightIndices.push(idx);
    }
  }

  if (bestLeftIndices.length === 0 || bestRightIndices.length === 0) {
    return { isLeaf: true, value: mean };
  }

  // Record importance
  featureImportances[bestFeature] += bestGain;

  const left = buildDecisionTree(
    X,
    y,
    bestLeftIndices,
    depth + 1,
    maxDepth,
    minSamplesLeaf,
    featureImportances,
    featureSubsetSize,
    rng
  );
  const right = buildDecisionTree(
    X,
    y,
    bestRightIndices,
    depth + 1,
    maxDepth,
    minSamplesLeaf,
    featureImportances,
    featureSubsetSize,
    rng
  );

  return {
    isLeaf: false,
    featureIdx: bestFeature,
    threshold: bestThreshold,
    left,
    right,
  };
}

function predictTree(node: TreeNode, x: number[]): number {
  if (node.isLeaf) return node.value;
  if (x[node.featureIdx] <= node.threshold) {
    return predictTree(node.left, x);
  }
  return predictTree(node.right, x);
}

export function trainRegressor(
  rawRows: Record<string, any>[],
  columns: string[],
  target: string,
  features: string[],
  testSize: number = 0.2,
  nEstimators: number = 100,
  filename: string = 'dataset',
  sizeBytes: number = 0
): MLResult {
  if (!target || !columns.includes(target)) {
    throw new Error('Choose a target column that exists in the dataset.');
  }
  if (!Array.isArray(features) || features.length === 0) {
    throw new Error('Select at least one input feature.');
  }
  if (new Set(features).size !== features.length) {
    throw new Error('Duplicate input features are not allowed.');
  }
  if (features.includes(target)) {
    throw new Error('The target column cannot also be an input feature.');
  }
  for (const f of features) {
    if (!columns.includes(f)) {
      throw new Error('One or more selected input features do not exist in the dataset.');
    }
  }
  if (testSize < 0.1 || testSize > 0.4) {
    throw new Error('Test proportion must be between 0.10 and 0.40.');
  }
  if (!Number.isInteger(nEstimators) || nEstimators < 10 || nEstimators > 500) {
    throw new Error('Number of trees must be an integer between 10 and 500.');
  }

  // Filter usable target rows
  type RowInfo = { originalRowNumber: number; row: Record<string, any>; targetVal: number };
  const usableRows: RowInfo[] = [];
  let targetRowsRemoved = 0;

  for (let i = 0; i < rawRows.length; i++) {
    const r = rawRows[i];
    const val = r[target];
    const num = typeof val === 'number' ? val : Number(String(val).trim().replace(/,/g, ''));
    if (val !== null && val !== undefined && val !== '' && !isNaN(num) && isFinite(num)) {
      usableRows.push({
        originalRowNumber: i + 2, // 1-indexed plus header row
        row: r,
        targetVal: num,
      });
    } else {
      targetRowsRemoved++;
    }
  }

  if (usableRows.length < 10) {
    throw new Error('At least 10 rows with a numeric target are required for training and evaluation.');
  }

  const uniqueTargetValues = new Set(usableRows.map((u) => u.targetVal));
  if (uniqueTargetValues.size < 2) {
    throw new Error('The target is constant; regression evaluation is not meaningful.');
  }

  const testCount = Math.max(2, Math.ceil(usableRows.length * testSize));
  const trainCount = usableRows.length - testCount;
  if (testCount < 2 || trainCount < 5) {
    throw new Error(
      'The selected split leaves too few training or test observations; provide more usable rows or choose a smaller test proportion.'
    );
  }

  // Categorize features into numeric and categorical
  const numericFeatures: string[] = [];
  const categoricalFeatures: string[] = [];

  for (const col of features) {
    let validNumeric = 0;
    let nonMissing = 0;
    const valuesSet = new Set<any>();

    // Check leakage: does feature exactly duplicate target?
    let duplicatesTarget = true;

    for (const u of usableRows) {
      const v = u.row[col];
      if (v !== null && v !== undefined && v !== '') {
        nonMissing++;
        valuesSet.add(v);
        const num = typeof v === 'number' ? v : Number(String(v).trim().replace(/,/g, ''));
        if (!isNaN(num) && isFinite(num)) {
          validNumeric++;
          if (num !== u.targetVal) {
            duplicatesTarget = false;
          }
        } else {
          duplicatesTarget = false;
        }
      } else {
        duplicatesTarget = false;
      }
    }

    if (duplicatesTarget && nonMissing === usableRows.length) {
      throw new Error(`Feature '${col}' duplicates the target values and would leak the answer.`);
    }

    if (nonMissing > 0 && validNumeric >= nonMissing * 0.8) {
      if (valuesSet.size > 1) {
        numericFeatures.push(col);
      }
    } else {
      if (valuesSet.size > 1) {
        categoricalFeatures.push(col);
      }
    }
  }

  const usableFeatures = [...numericFeatures, ...categoricalFeatures];
  const excludedFeatures = features.filter((f) => !usableFeatures.includes(f));

  if (usableFeatures.length === 0) {
    throw new Error('Selected input features are empty, missing, or constant. Choose at least one varying feature.');
  }

  // Deterministic train / test split with random_state = 42
  const rngSplit = mulberry32(42);
  const allIndices = Array.from({ length: usableRows.length }, (_, i) => i);
  // Fisher-Yates shuffle
  for (let i = allIndices.length - 1; i > 0; i--) {
    const j = Math.floor(rngSplit() * (i + 1));
    const temp = allIndices[i];
    allIndices[i] = allIndices[j];
    allIndices[j] = temp;
  }

  const testIndices = allIndices.slice(0, testCount);
  const trainIndices = allIndices.slice(testCount);

  // Fit preprocessing only on training set
  // 1. Numeric medians
  const numericMedians: Record<string, number> = {};
  for (const col of numericFeatures) {
    const trainVals: number[] = [];
    for (const tIdx of trainIndices) {
      const v = usableRows[tIdx].row[col];
      const num = typeof v === 'number' ? v : Number(String(v).trim().replace(/,/g, ''));
      if (v !== null && v !== undefined && v !== '' && !isNaN(num) && isFinite(num)) {
        trainVals.push(num);
      }
    }
    trainVals.sort((a, b) => a - b);
    numericMedians[col] = trainVals.length > 0 ? trainVals[Math.floor(trainVals.length / 2)] : 0;
  }

  // 2. Categorical categories (one-hot encoding)
  const categoryMaps: Record<string, string[]> = {};
  for (const col of categoricalFeatures) {
    const catCounts: Record<string, number> = {};
    for (const tIdx of trainIndices) {
      const v = usableRows[tIdx].row[col];
      const cat = v !== null && v !== undefined && v !== '' ? String(v) : '(missing)';
      catCounts[cat] = (catCounts[cat] || 0) + 1;
    }
    // Filter categories with frequency >= 2, max 500
    const sortedCats = Object.entries(catCounts)
      .filter(([_, count]) => count >= 2)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 500)
      .map(([cat]) => cat);
    categoryMaps[col] = sortedCats;
  }

  // Construct transformed feature vector
  // Mapping each transformed column back to its owner feature name
  const transformedFeatureOwners: string[] = [];

  for (const col of numericFeatures) {
    transformedFeatureOwners.push(col);
  }
  for (const col of categoricalFeatures) {
    for (const cat of categoryMaps[col]) {
      transformedFeatureOwners.push(col);
    }
  }

  function transformRow(row: Record<string, any>): number[] {
    const vec: number[] = [];
    // Numeric
    for (const col of numericFeatures) {
      const v = row[col];
      const num = typeof v === 'number' ? v : Number(String(v).trim().replace(/,/g, ''));
      if (v !== null && v !== undefined && v !== '' && !isNaN(num) && isFinite(num)) {
        vec.push(num);
      } else {
        vec.push(numericMedians[col] || 0);
      }
    }
    // Categorical One-Hot
    for (const col of categoricalFeatures) {
      const v = row[col];
      const cat = v !== null && v !== undefined && v !== '' ? String(v) : '(missing)';
      for (const validCat of categoryMaps[col]) {
        vec.push(cat === validCat ? 1 : 0);
      }
    }
    return vec;
  }

  const X: number[][] = usableRows.map((u) => transformRow(u.row));
  const y: number[] = usableRows.map((u) => u.targetVal);

  const numTransformedFeatures = X[0].length;
  const featureSubsetSize = Math.max(1, Math.floor(Math.sqrt(numTransformedFeatures)));
  const treeFeatureImportances = new Float64Array(numTransformedFeatures);

  // Train Random Forest
  const rngForest = mulberry32(42);
  const trees: TreeNode[] = [];

  for (let t = 0; t < nEstimators; t++) {
    // Bootstrap sampling from trainIndices
    const bootstrapIndices: number[] = [];
    for (let i = 0; i < trainIndices.length; i++) {
      const rIdx = Math.floor(rngForest() * trainIndices.length);
      bootstrapIndices.push(trainIndices[rIdx]);
    }

    const tree = buildDecisionTree(
      X,
      y,
      bootstrapIndices,
      0,
      12, // maxDepth
      2, // minSamplesLeaf
      treeFeatureImportances,
      featureSubsetSize,
      rngForest
    );
    trees.push(tree);
  }

  // Predict on held-out test rows
  type PredItem = { row: number; actual: number; predicted: number; absolute_error: number };
  const allPredictions: PredItem[] = [];

  let sse = 0;
  let sae = 0;
  let testSum = 0;

  for (const tIdx of testIndices) {
    const actual = y[tIdx];
    testSum += actual;
    const xVec = X[tIdx];

    let predSum = 0;
    for (let i = 0; i < nEstimators; i++) {
      predSum += predictTree(trees[i], xVec);
    }
    const predicted = predSum / nEstimators;
    const absErr = Math.abs(actual - predicted);

    sae += absErr;
    sse += Math.pow(actual - predicted, 2);

    allPredictions.push({
      row: usableRows[tIdx].originalRowNumber,
      actual: actual,
      predicted: predicted,
      absolute_error: absErr,
    });
  }

  allPredictions.sort((a, b) => a.row - b.row);

  const testN = testIndices.length;
  const mae = sae / testN;
  const rmse = Math.sqrt(sse / testN);

  // R² score
  const testMean = testSum / testN;
  let totalSS = 0;
  for (const tIdx of testIndices) {
    totalSS += Math.pow(y[tIdx] - testMean, 2);
  }
  const r2 = totalSS > 1e-10 && testN >= 2 ? 1 - sse / totalSS : null;

  // Feature importance grouped by original column
  const groupedImportance: Record<string, number> = {};
  for (const col of usableFeatures) {
    groupedImportance[col] = 0;
  }

  let totalImp = 0;
  for (let i = 0; i < numTransformedFeatures; i++) {
    const owner = transformedFeatureOwners[i];
    groupedImportance[owner] += treeFeatureImportances[i];
    totalImp += treeFeatureImportances[i];
  }

  const importanceList = Object.entries(groupedImportance)
    .map(([feature, val]) => ({
      feature,
      importance: totalImp > 0 ? round(val / totalImp, 6) : 0,
    }))
    .sort((a, b) => b.importance - a.importance);

  // Predictions CSV
  const csvLines = ['row,actual,predicted,absolute_error'];
  for (const p of allPredictions) {
    csvLines.push(`${p.row},${p.actual},${p.predicted.toFixed(3)},${p.absolute_error.toFixed(3)}`);
  }
  const predictionsCsv = csvLines.join('\n');

  // Warnings
  const warnings = [
    'This is ordinary regression using a reproducible random split, not a future forecast.',
    'Feature importance describes model reliance, not causation.',
    'Performance is measured only on the held-out test rows and may vary with other samples.',
  ];

  if (excludedFeatures.length > 0) {
    warnings.push(`Constant or entirely missing input fields were excluded: ${excludedFeatures.join(', ')}.`);
  }

  const dateCandidates = columns.filter((col) =>
    ['date', 'time', 'month', 'year'].some((token) => col.toLowerCase().includes(token))
  );
  if (dateCandidates.length > 0) {
    warnings.push(
      'Date-like columns exist. A random split can mix earlier and later records; use chronological evaluation for forecasting.'
    );
  }

  if (usableRows.length < 50) {
    warnings.push('A small dataset can make held-out metrics unstable.');
  }

  if (testIndices.length < 10) {
    warnings.push('The test set is small; R² and other metrics may be unreliable.');
  }

  if (r2 === null) {
    warnings.push('R² is omitted because test targets do not vary enough to make it meaningful.');
  } else if (r2 < 0) {
    warnings.push('Negative test R² means this model performed worse than predicting the test-set mean for these rows.');
  }

  const mlResult: MLResult = {
    filename,
    target,
    features: usableFeatures,
    excluded_constant_features: excludedFeatures,
    configuration: {
      model: 'RandomForestRegressor',
      n_estimators: nEstimators,
      test_size: testSize,
      random_state: 42,
      split: 'random',
    },
    counts: {
      usable_rows: usableRows.length,
      target_rows_removed: targetRowsRemoved,
      training_rows: trainIndices.length,
      test_rows: testIndices.length,
    },
    metrics: {
      mae,
      rmse,
      r2,
    },
    feature_importance: importanceList,
    predictions: allPredictions.slice(0, 250),
    prediction_count: allPredictions.length,
    predictions_csv: predictionsCsv,
    warnings,
    report: '',
  };

  const statistics = analyzeDataset(rawRows, columns, filename, sizeBytes);
  mlResult.report = renderReport(statistics) + '\n' + renderMLSection(mlResult);

  return mlResult;
}
