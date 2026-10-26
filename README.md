# InsyT

**Your data. Your insights. Smarter decisions.**

InsyT analyzes CSV and Excel spreadsheets and turns their contents into dataset profiles, evidence-backed findings, visualizations, quality checks, and Markdown reports. An optional Random Forest regression workflow evaluates a user-selected target on held-out data.

## Features

- CSV/XLSX upload with 10 MB, 100,000-row, and 100-column limits.
- Dataset profiling, KPIs, missingness, duplicate counts, and data quality observations.
- Data-derived time-series, category, and numeric-distribution charts.
- IQR outlier detection and Pearson correlation summaries.
- Evidence-backed findings, executive summary, and Markdown report download.
- Optional Random Forest regression with train-only preprocessing, held-out MAE/RMSE/R², feature importance, test predictions, and CSV export.
- One-click synthetic demo dataset, served as a frontend static asset and analyzed through the regular backend upload endpoint.
- Optional Gemini explanations; statistical analysis and reports work without an API key.

## Demo dataset

`sample_data/sample_sales.csv` contains 200 **synthetic** retail rows and the columns `Order_Date`, `Product`, `Region`, `Sales_Channel`, `Units_Sold`, `Revenue_INR`, `Discount_Percent`, and `Customer_Rating`. It includes missing cells, two deliberate duplicate records, and unusual numeric observations. The same CSV is mirrored at `frontend/public/sample_sales.csv` so Vite/Vercel can serve it at `/sample_sales.csv`. The **Try Demo Dataset** action sends it through `POST /api/analyze`; **Download Sample CSV** downloads the file.

## Technology and architecture

- Frontend: React, Vite, TypeScript, Tailwind CSS v4, Recharts, Lucide React, Axios.
- Backend: Python, FastAPI, pandas, NumPy, scikit-learn, openpyxl.
- Optional AI: Google Gen AI SDK (Gemini).

The browser posts files to `POST /api/analyze` and model requests to `POST /api/ml/train`. Uploaded files and models are kept in memory only. The ML endpoint fits imputers and encoders on training rows and returns held-out evaluation results; it is ordinary random-split regression, not forecasting.

## Local development

Python 3.10+ and Node.js 20+ are recommended. From the repository root, start the backend:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
uvicorn backend.main:app --reload
```

In another PowerShell window, start the frontend:

```powershell
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173/`. The API health endpoint is `http://127.0.0.1:8000/api/health`.

## Configuration

- `VITE_API_URL`: frontend build-time API origin; defaults to `http://localhost:8000` for local development. Set this to the deployed backend URL in the frontend hosting environment.
- `CORS_ORIGINS`: comma-separated exact browser origins allowed by FastAPI. Local defaults allow ports 5173 on localhost and 127.0.0.1. Set the exact deployed frontend origin in the backend hosting environment.
- `GEMINI_API_KEY`: optional backend-only secret. Never set this as a `VITE_` variable.
- `GEMINI_MODEL`: optional backend model name.

Use `backend/.env.example` and `frontend/.env.example` as local references. `.env` files are ignored by Git.

## Tests and build

From the repository root:

```powershell
python -m pytest backend/tests -q
cd frontend
npm ci
npm run build
```

## Deployment

The repository includes `vercel.json` for the static Vite frontend and `render.yaml` for the FastAPI service. No production deployment has been verified yet; no public URLs are listed until they are live and checked.

1. Push this repository to GitHub.
2. In Render, create a Blueprint from the repository and apply `render.yaml`. Set the requested `CORS_ORIGINS` value to the eventual Vercel origin, with no trailing slash. Render will provide the backend URL; verify `<backend-url>/api/health`.
3. In Vercel, import the repository with the repository root as the project root. Set `VITE_API_URL` to the Render backend URL, without a trailing slash, and deploy.
4. Update Render `CORS_ORIGINS` with the actual Vercel domain and redeploy if that domain was not known during initial service creation.
5. Verify the homepage, demo upload, CSV download, API health, normal CSV/XLSX uploads, report, and ML flow.

The free Render web service can spin down after inactivity and take about a minute to wake. The frontend upload requests allow up to 120 seconds. See [Render free service limitations](https://render.com/docs/free), [Render FastAPI deployment](https://render.com/docs/deploy-fastapi), and [Vercel Vite deployment](https://vercel.com/docs/frameworks/frontend/vite).

### Deployment status

- Frontend URL: not deployed.
- Backend health URL: not deployed.
- Gemini is optional and is not required for analysis.

## Limitations

- Excel import reads the first worksheet.
- Trends and category comparisons are descriptive; correlation and feature importance do not establish causation.
- Random train/test splits are not suitable for forecasting temporal data.
- Free backend hosting may cold-start after idle periods.
