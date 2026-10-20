# Runbooks

Each alert and each pipeline step links here. Every section says what the alert means, what to
check first, and how to recover. Commands assume the project image or a checkout with
`pip install -r requirements.txt && pip install -e .`.

## Stale or missing delivery
*Alert `FsapDeliveryStale`, task `check_delivery` failing.*

The weekly campaign results have not arrived, so plans are based on old data.

1. Check the landing prefix for this week's partition and the `_READY` marker.
2. If files are there but the marker is missing, the upstream export is still running or failed
   half way: ask the campaign-system on-call; do not create the marker by hand.
3. `check_delivery` retries six times, 30 minutes apart. Plans from the previous run stay valid
   meanwhile; tell the buying team in the channel if the delay reaches campaign cut-off.

## Contract breach or high quarantine
*Task `ingest` failed with exit code 3, or alert `FsapQuarantineHigh`.*

More rows than the contract allows broke it. Silver data from the last good run is untouched.

1. Open `reports/ingest/latest.json`: `top_reasons` says which rule broke.
2. Look at the rows: `quarantine/<dataset>/` keeps them with a `violations` column.
3. Upstream bug (e.g. negative units, prices above RRP): raise it with the owning team (`owner` in
   the contract) and wait for a corrected delivery. Do not loosen the contract to make the run pass.
4. Legitimate change (new category, new field): change the contract YAML, bump the version (major
   for breaking changes — CI enforces this), get it reviewed, rerun from `ingest`.

## dbt tests failing
*Task `dbt_build`.*

A model test or a model contract failed, e.g. daily sales no longer reconcile with offer totals.
Run `dbt build --select <model>+` locally against the same silver data, read the failing test's
SQL in `target/compiled`, and fix at the source (the ingest contract or the model), not by
deleting the test.

## Training gate rejected the model
*`train` logs "gate rejected version N".*

Not an outage: planning continues with the current champion. Read `reports/training/latest.json`:

- `unbiased` false: demand level moved (season change, campaign mix). Check the bias by fold; if
  only the last fold is off, wait a week for the calibration window to catch up.
- `calibrated` false: intervals too narrow or wide; check the dispersion against earlier versions.
- `significantly_better` false on a new recipe: the change did not help; do not force it.

The first ever model being rejected stops the DAG (exit code 5), because there is nothing to plan with.

## Forecast drift
*Alert `FsapForecastBiased`, monitoring group `outcomes`.*

Realised demand on planned campaigns differs from the forecast. Compare by category and brand in
`reports/monitoring/latest.json` and the dbt mart `mart_campaign_kpis`. A new brand or a traffic
change (marketing push, app outage) explains most cases. If it persists for two runs, retrain with
the latest data (`fsap train`) and let the gate decide.

## Monitoring checks
*Alert `FsapMonitoringChecksBreached`.*

`reports/monitoring/latest.json` lists every check with value, threshold and group. Drift checks
(PSI above 0.2) on a single feature usually come from an upstream definition change; compare
the feature's distribution in `gold/scoring_features` with the training window.

## API errors
*Alert `FsapApiErrors`.*

1. `kubectl -n assortment logs deploy/fsap-api` — every log line carries the request id returned
   to the caller in `x-request-id`.
2. `/ready` returns 503 when no champion model or no planning data is loaded: check that the
   batch run finished and the volume is mounted.
3. Roll back with `kubectl -n assortment rollout undo deploy/fsap-api` if a release caused it.

## Slow replanning
*Alert `FsapReplanSlow`.*

The optimiser has a 20-second limit per solve; campaigns with many more candidates than usual
take longer. Check `fsap_plans_total{status=~"limit.*"}`. Scale the API (`hpa`) if CPU-bound.

## GenAI guardrails
*Alert `FsapCopyViolations`, or a spike in `fsap_genai_guardrail_events_total`.*

The model is producing text or attributes the guardrails reject (overclaimed discounts, brands not
in the campaign, invented materials). Nothing unsafe is published — the template or lexicon
fallback is used — but quality drops. Check whether a prompt, model version or provider changed;
run `fsap curate` and compare `reports/genai/latest.json` with the previous report. Roll back the
provider settings (`FSAP_LLM_*`) if needed.
