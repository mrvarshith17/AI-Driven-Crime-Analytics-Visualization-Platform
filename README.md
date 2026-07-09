# KSP Crime Intelligence Platform — Setup Guide

## Step 1 — Generate data & train models (Kaggle)
1. Create a new Kaggle Notebook, upload `ksp_crime_intel_training.ipynb` (or paste its cells in).
2. Run all cells top to bottom (CPU only, no GPU/internet needed — everything is generated in-notebook).
3. Takes ~1-2 minutes. It will produce an `Output → artifacts/` folder containing:

   **Raw schema tables** (mirrors the real KSP FIR ER diagram — CaseMaster, ComplainantDetails, Victim,
   Accused, ArrestSurrender, ActSectionAssociation, ChargesheetDetails, District, Unit, CrimeHead,
   CrimeSubHead, Act, Section, Court, Employee). Include these in your repo/demo to show schema fidelity —
   the app doesn't read them directly.

   **Flattened analytics view + ML artifacts** (what `app.py` actually reads):
   - `crime_incidents.csv` — 20,000 cases flattened from CaseMaster + joins, one row per case
   - `district_daily_risk.csv` — district-day aggregates used for risk scoring
   - `districts_reference.csv` — district geo + socio-economic reference table
   - `hotspot_clusters.csv` — DBSCAN spatiotemporal cluster summary
   - `graph_edges.csv` / `graph_nodes.csv` — accused-victim-police station relationship graph
   - `top_repeat_offenders.csv` — highest-degree (most connected) accused persons
   - `isolation_forest.pkl` — trained anomaly-detection model
   - `risk_model.pkl` — trained RandomForest risk-scoring model
4. On Kaggle: click the "Output" tab on the right panel → download the whole `artifacts` folder.

## Step 2 — Run the Streamlit app locally
```bash
mkdir ksp-crime-platform
cd ksp-crime-platform
# put app.py + requirements.txt here
# put the downloaded artifacts/ folder here too, so structure is:
#   ksp-crime-platform/
#     app.py
#     requirements.txt
#     artifacts/
#       crime_incidents.csv
#       ...

pip install -r requirements.txt
streamlit run app.py
```

## What each dashboard page does
| Page | Maps to slide feature |
|---|---|
| Overview | High-level KPIs, trend line, AI insights, and red-zone Emerging Trend Alerts |
| Search & AI Query | Natural-language query + global search across records |
| Geospatial Hotspot Map | "Advanced Visualization" — district-level interactive maps + DBSCAN hotspot clusters |
| District Drill-down | "District-Level Drill-down" |
| Network & Link Analysis | "Criminological Network & Link Analysis" — relationship mapping, repeat offender tracking, and Association Detection (Act/Crime association rules) |
| Predictive Risk Dashboard | "Sociological & AI-Driven Predictive Dashboards" — risk scoring + explainability |
| Anomaly Detection | "Anomaly Detection" visual call-outs |
| Trend & Pattern Discovery | "Pattern & Trend Discovery" — MO breakdown, 30-day forecast, socio-economic correlation, full Emerging Trend Alert log |

## Notes for your demo
- **Data**: This uses synthetic data generated to populate the actual KSP FIR System ER schema (CaseMaster, Victim, Accused, ArrestSurrender, Act/Section, CrimeHead/CrimeSubHead, etc.) — real KSP incident records aren't public, but the schema itself is real. Say this explicitly in your pitch: "synthetic data populated against the real KSP database design, since raw records are confidential." That's a stronger claim than a made-up schema and shows you actually engineered against the provided ER diagram.
- `CrimeNo` values follow the exact spec from the ER doc: 1-digit category code + 4-digit District ID + 4-digit Unit ID + 4-digit year + 5-digit serial (e.g. `144130075202500001`).
- **No GPU needed anywhere** — IsolationForest, RandomForest, and DBSCAN are all fast CPU models; training takes seconds on 20k rows.
- **Swapping in real data later**: if you get an anonymized real dataset, keep the same column names (`district`, `crime_type`, `latitude`, `longitude`, `date`, `hour`, `suspect_id`, `victim_id`, `severity_score`, etc.) and the notebook + app work unmodified.
- To deploy publicly for judges, push this folder to GitHub and deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud) — point it at `app.py`, it installs `requirements.txt` automatically. Just make sure `artifacts/` is committed to the repo (the pkl/csv files are small, well under GitHub limits).
