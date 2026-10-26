# InsyT

**Your data. Your insights. Smarter decisions.**

InsyT analyzes CSV and Excel spreadsheets and turns their contents into dataset profiles, evidence-backed findings, visualizations, quality checks, and downloadable Markdown reports. An integrated Random Forest regression workflow evaluates a user-selected target on held-out data with train-only preprocessing, feature importance ranking, and prediction exports.

## Features

- **Spreadsheet Processing**: Drag-and-drop or browse CSV and XLSX spreadsheets with up to 10 MB, 100,000-row, and 100-column bounds.
- **Dataset Profiling & Quality Checks**: Automated type inference (numeric, date, categorical), null counts, duplicate detection, constant column flags, and cardinality checks.
- **Statistical Analytics & Visualizations**:
  - Metric KPIs (record count, field count, numeric means and medians, category cardinality).
  - Frequency distribution histograms for numeric fields.
  - Time-series trend lines with adaptive period aggregation (daily, weekly, monthly).
  - Category composition bar charts.
- **Outlier & Correlation Detection**:
  - 1.5×IQR outlier detection with lower/upper thresholds and context for each flag.
  - Pearson correlation matrix with strength classification and relationship summaries.
- **Evidence-Backed Findings**: Ranked data observations with supporting JSON evidence, methodologies, and actionable next steps.
- **Predictive Intelligence**:
  - Leakage-conscious Random Forest regression.
  - Train/test splits (20%, 25%, 30%, 40%) with held-out MAE, RMSE, and R² evaluation.
  - Actual vs. Predicted scatter plot and feature importance rankings.
  - Predictions CSV and full markdown model report exports.
- **Interactive Sample Dataset**: Instant one-click exploration using synthetic 200-row retail data with deliberate missing cells, duplicates, and statistical outliers.

## Technology Stack

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS v4 & custom design tokens
- **Visualizations**: Recharts
- **Parsing**: PapaParse (CSV) and SheetJS (XLSX)
- **Icons**: Lucide React

## Development

Install dependencies and start the local development server:

```bash
npm install
npm run dev
```

Build for production:

```bash
npm run build
```

Verify code quality:

```bash
npm run lint
```
