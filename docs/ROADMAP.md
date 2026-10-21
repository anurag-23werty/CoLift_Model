# Roadmap

The project takes about 8–10 weeks part-time. Each phase ends with something demoable and a git tag.

## Phase 0: Scaffold (Week 1)
- `pyproject.toml` (Python 3.11, uv), `Makefile`, ruff + mypy + pre-commit
- `src/uplift` package layout, `tests/`, empty GitHub Actions CI that runs lint + tests
- **Done when:** CI is green on an empty package.

## Phase 1: Data and EDA (Week 1–2)
- Ingestion via `sklift.datasets` (Hillstrom, Criteo) to parquet, then `dvc add`
- Pandera schemas, SRM check, missing-value and type checks
- EDA notebook: treatment balance, outcome rates per arm, **naive ATE with CI**
- **Done when:** `make data` reproduces the processed data from scratch, and the ATE matches the published numbers.

## Phase 2: Models (Week 2–4)
- Baselines: random ranking, a propensity model (to show why it fails)
- S, T, X and DR learners with LightGBM base models; causal forest (EconML)
- Hyperparameter search with Optuna, logged to MLflow
- **Done when:** every model is logged in MLflow with params, metrics and artifacts.

## Phase 3: Evaluation and policy (Week 4–5)
- Qini / uplift curves, AUUC, uplift@k, decile calibration, bootstrap CIs
- Policy simulator: profit curve compared with blanket targeting, optimal k, budget constraint
- Segment analysis: persuadables / sure things / lost causes / sleeping dogs
- **Done when:** one command produces a report with the curves and a "₹ saved" headline.

## Phase 4: MLOps backbone (Week 5–6)
- `dvc.yaml` pipeline (download → validate → split → train → evaluate → register)
- MLflow Model Registry with `@champion` / `@challenger` aliases, plus a quality gate
- Model card generated per registered version
- **Done when:** `dvc repro` rebuilds everything, and only models that pass the gate get promoted.

## Phase 5: Serving (Week 6–7)
- FastAPI app with Pydantic schemas, `/score`, `/batch_score`, `/policy`, `/health`
- Dockerfile (multi-stage, slim), unit and API tests, load test with Locust
- **Done when:** `docker run` serves predictions from the registry model with p95 < 100 ms.

## Phase 6: Frontend and deployment (Week 7–8)
- Streamlit simulator: sliders for coupon cost, margin and budget; live profit curve; single-user scoring
- Deploy the API (Render or HF Docker Space) and the app (HF Spaces), and set up CD from GitHub Actions
- **Done when:** a public URL works from a phone.

## Phase 7: Monitoring and retraining (Week 8–9)
- Prediction logging, Evidently drift report published as a static HTML page
- Weekly GitHub Actions cron: drift check, then retrain and evaluate the challenger, then promote if it's better
- Simulated holdout report showing realised lift (model-targeted vs random)
- **Done when:** the drift report is visible in the dashboard and the retrain workflow has run end to end once.

## Phase 8: Scale and write-up (Week 9–10)
- Full Criteo 14M run: training time, memory and Qini per model
- Blog post / Medium article; 2-minute demo video (Loom); README results table filled in
- **Done when:** the README has real numbers, and the demo link and video are live.

## Stretch goals
- Multi-treatment (choose *which* coupon: 10% off vs ₹50 off vs free delivery)
- Budget-constrained optimisation (knapsack over users and coupons)
- Uplift uncertainty: target only users whose CI lower bound is > 0
- Off-policy evaluation of the learned policy (IPW / doubly robust policy value)
