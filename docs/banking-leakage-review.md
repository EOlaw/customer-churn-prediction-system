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

- Stratified random 60/20/20 (seed 42), fixed and recorded as `split` in every training report with partition sizes and SHA-256 fingerprints of the sorted customer IDs per partition. Training asserts the partitions are disjoint and cover every row.
- Preprocessing (median imputation, scaling, one-hot) lives inside the pipeline and is fitted on training rows only. Defect fixed: both candidate pipelines previously shared a single preprocessor object; each now gets its own clone.
- The test partition is scored once for the validation-selected model. With two candidates the selection effect is small but nonzero.
- Rows duplicating another row's features and label under a different ID are reported (`duplicate_feature_rows`) because a random split can place copies in both train and test and inflate scores.
- A random split is only valid for exchangeable, one-row-per-customer data. It says nothing about drift across time.

## 6. Open leakage-adjacent risks

- `score-all` scores every imported customer, including the 60% used to train and the 20% used for selection. Dashboard score summaries therefore mix in-sample and held-out rows and are optimistic. Partition membership is fingerprinted but not stored per customer.
- The synthetic generator draws `Exited` from a formula of the features, so synthetic metrics reflect that formula, not any banking behavior.
