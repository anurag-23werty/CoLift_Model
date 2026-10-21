# Architecture

This document describes how the uplift targeting system is put together: the offline training plane, the online
serving plane, and the feedback loop that keeps the model honest after deployment.

## 1. Design principles

1. **Uplift, not propensity.** Every model, metric and dashboard number is about *incremental* outcome.
2. **Always keep a random holdout.** In production, 5–10% of eligible users are randomised into treatment and control regardless of the model.
   That is the only way to (a) measure true incremental lift and (b) collect unbiased training data for the next model.
   A model trained only on its own targeted users learns from a biased sample.
3. **Batch first, real-time second.** Coupon campaigns are planned daily or weekly, so the main path is a batch scorer. The REST API exists for
   on-demand scoring and for the demo.
4. **Reproducibility.** Any registered model can be traced back to a git commit, a DVC data hash and an MLflow run.
5. **Free-tier deployable.** Every component runs on free hosting so the demo stays online for recruiters.

## 2. System context

```mermaid
flowchart TB
    subgraph Offline["Offline / training plane"]
        RAW[(Raw data<br/>Hillstrom · Criteo)] --> ING[Ingest & validate]
        ING --> FEAT[Feature pipeline]
        FEAT --> DS[(Processed datasets<br/>DVC-versioned)]
        DS --> TR[Trainer<br/>S/T/X/DR-learners · CausalForest]
        TR --> EV[Evaluator<br/>Qini · AUUC · calibration · bootstrap]
        EV --> GATE{Quality gate<br/>Qini ≥ champion − ε?}
        GATE -- pass --> REG[(MLflow Model Registry)]
        GATE -- fail --> X[Reject + alert]
    end

    subgraph Online["Online / serving plane"]
        REG -->|Production alias| API[FastAPI scoring service]
        REG -->|Production alias| BATCH[Batch campaign scorer]
        API --> UI[Streamlit campaign simulator]
        BATCH --> CAMP[(Campaign list<br/>user_id · uplift · treat flag)]
    end

    subgraph Feedback["Feedback & monitoring"]
        API --> LOGS[(Prediction logs)]
        CAMP --> HOLD[Random holdout split<br/>5–10% randomised]
        HOLD --> OUT[(Observed outcomes)]
        LOGS --> DRIFT[Evidently drift report]
        OUT --> LIFT[Realised lift report<br/>model-targeted vs random]
        DRIFT --> TRIG{Retrain?}
        LIFT --> TRIG
    end

    TRIG -- yes --> TR
```

## 3. Training pipeline (DVC stages / Prefect flow)

```mermaid
flowchart LR
    s1[download] --> s2[validate<br/>Pandera schema,<br/>treatment balance,<br/>SRM check]
    s2 --> s3[split<br/>stratify on T×Y<br/>70/15/15]
    s3 --> s4[featurize<br/>encoders fit on train only]
    s4 --> s5a[train_baselines]
    s4 --> s5b[train_meta_learners]
    s4 --> s5c[train_causal_forest]
    s5a & s5b & s5c --> s6[evaluate<br/>val set → model selection]
    s6 --> s7[final_test<br/>untouched test set, once]
    s7 --> s8[register<br/>MLflow + model card]
```

Notes:

- **SRM (sample ratio mismatch) check:** confirm the treatment/control ratio matches the design (1/3 per arm for Hillstrom, ~85/15 for Criteo). If it doesn't, the randomisation is suspect.
- **ATE sanity check:** the simple difference in means `mean(Y|T=1) − mean(Y|T=0)` must roughly match the mean predicted CATE.
- **Criteo scale strategy:** meta-learners with LightGBM train on the full 14M rows. Causal forest trains on a stratified 1–2M subsample, because it is memory-bound. The benchmark reports the cost/accuracy trade-off.
- **Hillstrom multi-treatment:** model *Men's email vs none* and *Women's email vs none* separately, or use EconML's multi-treatment support. The policy then picks the best arm per user.

## 4. Model layer

| Model | Idea | Strength | Weakness |
|---|---|---|---|
| S-learner | One model with T as a feature | Simple | Tree models often ignore T, so uplift shrinks to 0 |
| T-learner | Separate models for treated and control | Easy, strong baseline | Errors from the two models add up; weak when one arm is small (Criteo control = 15%) |
| X-learner | Imputes individual effects, weighted by propensity | Handles imbalanced arms well | More moving parts |
| DR-learner | Doubly robust pseudo-outcome regression | Low bias, principled | Needs cross-fitting |
| Causal Forest (DML) | Honest forest that splits on effect heterogeneity | Confidence intervals per user, interpretable | Slow and memory-heavy at 14M |

The randomised propensity is **known** (e.g. 0.85 on Criteo), so it is passed as a constant rather than estimated.

## 5. Serving

```mermaid
sequenceDiagram
    participant C as Client / Streamlit
    participant A as FastAPI
    participant M as Model loader (cached)
    participant R as MLflow Registry
    participant L as Prediction log

    A->>R: on startup: fetch models:/uplift@production
    R-->>A: model artifact + feature pipeline + version
    C->>A: POST /score {features}
    A->>A: Pydantic validation
    A->>M: predict_uplift(x)
    M-->>A: τ̂(x), CI (if causal forest)
    A->>A: policy rule: treat if τ̂·margin > expected discount cost
    A->>L: log(request_id, features, τ̂, decision, model_version)
    A-->>C: {uplift, segment, recommend_coupon, model_version}
```

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + loaded model version |
| POST | `/score` | Single user → uplift + decision |
| POST | `/batch_score` | File upload → ranked campaign list |
| POST | `/policy` | cost, margin, budget → optimal k, expected profit, profit curve |
| GET | `/metrics` | Prometheus-style counters (requests, latency) |

## 6. Deployment topology (free tier)

```mermaid
flowchart LR
    DEV[Developer laptop] -->|git push| GH[GitHub repo]
    GH --> CI[GitHub Actions<br/>ruff · mypy · pytest<br/>train smoke test]
    CI -->|on main| IMG[Build Docker images → GHCR]
    IMG --> RENDER[Render / HF Docker Space<br/>FastAPI container]
    GH -->|sync| HFS[Hugging Face Space<br/>Streamlit app]
    HFS -->|HTTPS| RENDER
    DAGS[DagsHub<br/>MLflow server + DVC remote] <-->|tracking / artifacts| CI
    DAGS -->|model pull| RENDER
    CRON[GitHub Actions cron<br/>weekly retrain + drift report] --> DAGS
```

## 7. Monitoring

| Signal | Tool | Action |
|---|---|---|
| Feature drift (PSI / KS per feature) | Evidently | Report + alert if PSI > 0.2 |
| Prediction drift (distribution of τ̂) | Evidently | Alert if the share of predicted persuadables shifts sharply |
| Realised incremental lift (holdout) | Custom report | Retrain if model-targeted lift < random-targeted lift |
| Latency / error rate | FastAPI middleware logs | Alert on p95 > 200 ms or 5xx > 1% |

## 8. Known risks

- **Post-treatment leakage**: Criteo `exposure` and anything else measured after the coupon must not be used as a feature.
- **Small effects, noisy metrics**: Criteo conversion is about 0.3%, so the differences between models need bootstrap CIs.
- **Feedback loop bias**: without the random holdout, retraining on targeted users degrades the model over time.
- **Offline ≠ online**: Qini on historical RCT data is a proxy. The real validation is an online A/B test of model targeting against blanket targeting.
