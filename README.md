# Flash-sale assortment platform

The data, ML and decision platform behind an online outlet's daily sale campaigns. Brands offer
lots of discounted articles; for every upcoming campaign the platform decides **which offers to
list and how many units to buy**, forecasts each article's sell-through, writes the campaign copy,
and tells the team when data, models or decisions go wrong.

It is built the way such a system runs in production: data contracts and Spark jobs on Databricks,
dbt models, an Airflow DAG, a model registry with a release gate, a decision API on Kubernetes,
and dashboards, alerts and runbooks.

```
landing ──► Spark ingest ──► silver ──► dbt marts ──► Spark features ──► train + gate ──► plan ──► curate ──► monitor
 (weekly    (contracts,       (typed,     (KPIs,        (point-in-time)    (MLflow,        (MILP)    (GenAI +    (data, drift,
  drops)     quarantine)       clean)      tests)                          champion)                  guardrails)  outcomes)
                                                                                              │
                                                    FastAPI decision service on Kubernetes ◄──┘  plans, what-if re-planning,
                                                    Prometheus metrics, Grafana dashboard        title curation, copy checks
```

All data is synthetic (see [Honest limits](#honest-limits)): real flash-sale data is not public,
so the platform runs against a simulator whose true demand is known. That makes it possible to
measure things real data never shows — how wrong sold-out sales are as a demand signal, and how
much profit a plan leaves on the table compared with perfect knowledge.

## What each part does

| Step | Runs on | What it does |
|---|---|---|
| `check_delivery` | pod | Waits for the week's delivery to be complete (`_READY` marker); retried six times. |
| `ingest` | Databricks job (Spark) | Landing → bronze (raw text + source file + ingest time) → silver, enforcing a **data contract** per dataset ([contracts](src/fsap/contracts/)). Bad rows go to a quarantine table with the rule they broke; if more than the contract allows are bad, the run stops and the last good silver stays in place. |
| `dbt_build` | pod (dbt-databricks; DuckDB locally) | Staging views and marts (`fct_offer_performance`, `mart_campaign_kpis`, `mart_brand_category_weekly`) with enforced model contracts and 18 data tests, including a reconciliation of daily sales against offer totals. |
| `features` | Databricks job (Spark) | Point-in-time features: brand, brand×category, category and article history from campaigns that **ended before** the target campaign started; brand fatigue from the campaign calendar; expected traffic. Sold-out offers are **uncensored** with the sales-by-day profile learned from offers that never sold out. |
| `train` | pod | Rolling-origin backtest (4 folds), LightGBM Tweedie forecaster with a negative-binomial distribution, recent-level and dispersion calibration, **release gate** (significantly better by paired bootstrap over campaigns, better in most folds, unbiased, calibrated), MLflow run + registered version with data fingerprint, contract versions and gate decision; promoted to `champion` only if it passes. |
| `plan` | pod | For each upcoming campaign a **mixed-integer program** (HiGHS) chooses offers and buy depth to maximise expected contribution under slots, buy budget, minimum fill and brand-share rules ([planning_rules.yaml](conf/planning_rules.yaml), owned by the business). Every plan line explains itself and records model version, rules digest and input fingerprint. |
| `curate` | pod | GenAI: supplier titles → clean attributes, and campaign copy in German and English, with guardrails (below). |
| `monitor` | pod | Data (quarantine, freshness), drift (PSI on key features) and outcome checks (forecast WAPE and bias, realised vs expected contribution on planned campaigns), each with a threshold and a runbook link. |

The [Airflow DAG](airflow/dags/flash_sale_assortment.py) runs these steps daily. With
`FSAP_RUNTIME=production` Spark steps become `DatabricksRunNowOperator`s against the jobs in
[the asset bundle](deploy/databricks/bundle.yml) and the rest become `KubernetesPodOperator`s from
the project image; with `local` every step is the same CLI command in the project's own virtualenv.
Airflow never imports the project, so the scheduler's dependencies never clash with the ML stack.

### Decision API

`uvicorn fsap.serving.app:create_app --factory` ([app.py](src/fsap/serving/app.py))

- `GET /v1/campaigns/{id}/plan` — the plan with a reason per offer ("listed: partial lot (expected profit peaks below the full lot)", "not listed: budget used up by better articles", …)
- `POST /v1/campaigns/{id}/replan` — what-if: a brand lowers its cost or the buyer drops an offer; changed offers are checked against the `candidate_offers` contract before re-solving
- `POST /v1/curation/attributes`, `POST /v1/curation/copy/check`
- `/metrics` for Prometheus, `/ready` fails until a champion model and planning data are loaded; every response carries `x-request-id`, logs are JSON

### GenAI with guardrails

- **Attributes.** A provider-agnostic client (Anthropic Messages API or any OpenAI-compatible
  endpoint) asks for JSON within a fixed schema. Nothing is trusted as-is: values outside the
  schema are rejected, every attribute must be **grounded** in a word of the title (fuzzy, to
  survive typos), and malformed or failed responses fall back to a deterministic lexicon
  extractor. Each event is counted and alerted on.
- **Copy.** Every text is validated before use: discount claims must be true for the listed
  articles and say "bis zu / up to" unless every article has that discount (EU price-indication
  rules), only brands in the campaign may be named, no vague environmental claims, no false
  urgency, channel length limits. A failing draft gets one retry with the violations as feedback,
  then the template is used.

## Results on the simulated outlet

Two years of history (3,333 campaigns, 227,655 offers), then 12 weeks of upcoming campaigns
(355 campaigns, 39,595 candidate offers) that the policies plan and the simulator plays out.

| | |
|---|---|
| Data contracts | 1,034,690 delivered rows; the 818 deliberately corrupted rows (negative units, sales above units bought, missing prices, unknown articles, prices above RRP, duplicates) were quarantined, and no clean rows were |
| Censoring | On sold-out offers, units sold understate true demand by 64%; the uncensored estimate is within 3% |
| Forecast backtest (4 folds, 499 campaigns, 32,853 offers) | WAPE 0.577 vs 0.634 for the history baseline: 5.6 points better (95% CI 4.1–6.9), better in 4 of 4 folds; bias +5.6%; the 80% interval covers 84% of uncensored outcomes. Against true demand: WAPE 0.60, where even the true mean scores 0.46 — the remaining gap is article and campaign noise |
| Planning (355 campaigns, same random demand for every policy) | Forecast + optimiser: €13.84M contribution vs €8.06M for a sell-through rule of thumb (**+72%**, 95% CI +64% to +81%), better in 99% of campaigns, with €5.1M less stock bought; sell-through 74% vs 56%. That is **84% of the gain an optimiser with perfect knowledge of demand achieves** (€14.94M). The same optimiser on the baseline forecast reaches +56%, so the better forecast adds 16 points |
| Plan accuracy | Expected contribution €13.06M, realised €13.63M on the planned campaigns (within 5%) |
| GenAI | 710 campaign texts, 0 rule violations; the validator catches 8 of 8 adversarial drafts with 0 false alarms. Against a simulated model that invents attributes and sometimes returns prose, the guardrails rejected all 43 invented values and handled all 36 malformed responses: 0 hallucinated attributes reached the catalogue |
| Pipeline | The Airflow DAG runs all 8 steps end to end in CI (`airflow dags test`) |
| Tests | 43 project tests (contracts in pandas and Spark agree row for row, point-in-time features, gate, optimiser constraints, LLM client against mocked Anthropic/OpenAI endpoints, the API end to end), 5 DAG integrity tests, 25 dbt models and tests |

Why the planning gain is so large: in the simulator, as in an outlet, many offers lose money once
returns, fulfilment and leftover stock are counted, and many lots are bigger than demand. The
optimiser's main wins are leaving those offers out and buying partial lots. Real gains depend on
how good the current buying process already is; the comparison here is against a reasonable rule,
not against experienced buyers.

## Run it

```bash
pip install -r requirements-dev.txt && pip install -e .     # Python 3.12+, Java 17+ for Spark
scripts/run_pipeline.sh                                      # every step once (~6 minutes)
uvicorn fsap.serving.app:create_app --factory                # decision API on :8000
python -m fsap.cli evaluate                                  # policies vs true demand (simulation only)
pytest                                                       # all tests, including Spark and end-to-end
```

Or everything in containers, with Prometheus and the Grafana dashboard:

```bash
docker compose up --build    # API :8000, Prometheus :9090, Grafana :3000
```

Deployment: [Dockerfile](Dockerfile) (non-root, runtime and local targets),
[Kubernetes](deploy/k8s/) (restricted pod security, probes, HPA, PDB, network policy, IRSA service
accounts, PrometheusRule), [Databricks bundle](deploy/databricks/bundle.yml),
[Terraform](deploy/terraform/main.tf) (ECR, KMS-encrypted S3 with lifecycle rules, IRSA roles),
[alerts](ops/prometheus/alerts.yml), [dashboard](ops/grafana/dashboards/fsap.json) and
[runbooks](docs/runbooks.md) — every alert links to a runbook section, and a test keeps it so.

## Honest limits

- **Synthetic data.** The demand model (traffic, discount response, cannibalisation, brand
  fatigue, returns) is plausible but invented, so the numbers show that the system works and how
  to measure it, not what it would earn at a real outlet.
- **The LLM path is tested, not benchmarked.** No API key was used: the client is tested against
  mocked Anthropic and OpenAI-compatible endpoints, and the guardrails against a simulated model
  with typical failure modes. The lexicon extractor scores well here because the simulated
  suppliers use its vocabulary; real supplier data has an open vocabulary, which is what the LLM
  path is for.
- **Production targets are validated, not deployed.** Databricks jobs, the Kubernetes manifests
  and Terraform are checked in CI (`terraform validate`, kubeconform, image builds and a container
  smoke test), not run against a real workspace or cluster. Locally Spark writes Parquet; on
  Databricks the same jobs would write Delta tables.
- Cannibalisation is handled by re-forecasting at the chosen number of articles until it settles;
  substitution between specific articles is not modelled.

## License

MIT
