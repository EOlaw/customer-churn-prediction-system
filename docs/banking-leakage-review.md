# Banking leakage review

Scope: `banking-account-exit-v1` on `banking-csv-contract/1.0.0`. This review reads the code and the data contract; it did not analyze the real public CSV (not available locally). Findings that depend on real data are marked **unverified**. No timestamps or prediction horizon are assumed anywhere.

## 1. Identifiers

- `CustomerId` is excluded from features, enforced by a test (features are an explicit allow-list in `ChurnDefinition`). It is stored only as `external_id` for lookup.
- `RowNumber` and `Surname` are ignored at ingestion and not stored. If a source orders rows by account-opening or exit order, `RowNumber` or the ID's magnitude could encode the label indirectly, so keeping them out matters. **Unverified** for the real file.
- Identifier uniqueness is now enforced (whitespace variants included), so one customer cannot land in both train and test under two spellings.

## 2. Target-derived fields

- Only `Exited` is the label. The contract **rejects** `Complain`, `Satisfaction Score`, `Card Type`, and `Point Earned`. These appear in related public variants of the dataset and are reported in public discussion to be outcome-adjacent (for example, a complaint flag recorded around the exit). This repository has not verified that claim. Verify against your copy's data dictionary before ever removing them from the deny-list.
- Unknown extra columns are excluded from storage and training and reported as warnings, so they cannot enter the model silently; features are an explicit allow-list.

## 3. Feature availability (all **unverified**; the file has no dates)

| Feature | Concern |
| --- | --- |
| `Geography`, `Gender` | Static, available early. Sensitive attributes: require a fairness and permitted-use review before any consequential use |
| `Age`, `CreditScore`, `EstimatedSalary` | Plausibly available before exit; measurement date unknown |
| `Tenure` | Unknown whether measured to the exit date (a mechanical function of when the customer left) or to the snapshot |
| `Balance`, `NumOfProducts`, `IsActiveMember`, `HasCrCard` | Account state may be recorded **after** a customer has wound down the relationship (zero balance, closed products, inactivity), which would make them consequences of exit rather than predictors |

If these were captured after the exit, model scores are retrospective classification of already-exited customers and not a usable early warning. The data cannot settle this. A prospective design needs a dated feature cutoff before a defined exit window.

## 4. Cross-sectional limitations

There is no observation date, feature cutoff, or horizon, so the report records `prediction_horizon: null`. The task is "which snapshot rows are labeled exited", not "who will exit". Scores on imported customers are retrospective, and nothing here supports calibration, forecast, or retention-impact claims. Class-weighted logistic regression also distorts probabilities, so Brier scores are diagnostics only.

## 5. Split strategy

- Stratified random 60/20/20 (seed 42) computed from customer IDs in sorted order (so file row order cannot change assignments), recorded as `split` in every training report with partition sizes and SHA-256 fingerprints of the customer IDs per partition, and stored per customer in the database at registration. Training asserts the partitions are disjoint and cover every row.
- Preprocessing (median imputation, scaling, one-hot) lives inside the pipeline and is fitted on training rows only. Defect fixed: both candidate pipelines previously shared a single preprocessor object; each now gets its own clone.
- Phase 3 replaces the earlier train/validation/test design with train / calibration / test. Candidate comparison uses cross-validation inside the training partition; the calibration partition fits the calibrator and threshold; the test partition is evaluated once (details in [the ML methodology](banking-ml-methodology.md)).
- Rows duplicating another row's features and label under a different ID are reported (`duplicate_feature_rows`) because a random split can place copies in both train and test and inflate scores.
- A random split is only valid for exchangeable, one-row-per-customer data. It says nothing about drift across time.

## 6. Leakage-adjacent risks

- **Resolved in Phase 3:** `score-all` scores every imported customer, including those used for training and calibration. Each customer's partition is now stored per training run (`split_assignments`), batch scores are labelled `retrospective_batch`, and the API and dashboard report scores per partition. Training and calibration scores are marked in-sample, no pooled mean score is shown, segment means use held-out test customers only, and the risk chart defaults to the test partition. Rows migrated from older schemas have no recorded split and are shown as "No split recorded · unverified".
- **Sensitivity analysis for uncertain availability (Phase 3):** the model is refit without `Tenure`, `Balance`, `NumOfProducts`, `IsActiveMember`, and `HasCrCard` (the features in section 3 that may be consequences of exit), and without `Gender`, `Age`, `Geography`. The comparison is reported with paired-bootstrap intervals and never used to select the model. A small drop does not prove the features are leak-free, and a large one does not prove they leak; only dated data can settle availability.
- Duplicate-feature rows can still straddle partitions (reported as `duplicate_feature_rows`).
- The synthetic generator draws `Exited` from a formula of the features, so synthetic metrics reflect that formula, not any banking behavior.
