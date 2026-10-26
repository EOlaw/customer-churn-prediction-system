# Banking ML methodology (Phase 3)

Scope: `banking-account-exit-v1`, contract `banking-csv-contract/1.0.0`. The task is **retrospective classification of `Exited` in a cross-sectional snapshot**. The file has no observation date, feature cutoff, or prediction horizon, so nothing here is a prospective churn forecast, and no result supports claims about retention impact or causes. The public banking CSV has not been analyzed; all implementation and tests use clearly labeled synthetic data, whose metrics describe generated data only.

## Data roles

| Partition | Share | Used for | Never used for |
| --- | --- | --- | --- |
| `train` | 60% | fitting candidates, cross-validated comparison and hyperparameter search, final refit | calibration, threshold, final metrics |
| `calibration` | 20% | choosing and fitting the probability calibrator; choosing the decision threshold | fitting the classifier, final metrics |
| `test` | 20% | one held-out evaluation | any choice |

Assignment is a stratified split of customers ordered by ID, so it is identical for the same IDs, labels, and seed regardless of file row order. It is verified disjoint and exhaustive, fingerprinted per partition (SHA-256 of sorted IDs), written to `<run>.splits.csv`, and stored per customer in `split_assignments` at registration.

Isolation is structural and tested: `fit_run(train, calibration, ...)` has no parameter for test data; `evaluate_run` is called only after fitting; a test spies on every model fit, calibrator fit, and threshold choice and asserts which rows each saw; another refits the calibrator from stored calibration rows alone and reproduces the stored test probabilities.

## Candidates and selection (training partition only)

| Candidate | Search |
| --- | --- |
| Dummy (class prior) | none; reference |
| Logistic regression | `C` in {0.1, 1, 10} |
| Gradient-boosted trees (`HistGradientBoostingClassifier`) | depth {2, 3} × learning rate {0.05, 0.1} × rounds {100, 200} (8 configurations) |

Boosted trees are included because age and product-count effects in retail-banking data are plausibly non-monotone and interactive, which a linear model cannot represent; the search is deliberately small for ~1k–10k rows. Scoring metric: average precision (PR-AUC), 5-fold stratified CV. Rule: the **simplest candidate whose mean is within one standard error of the best**. The CV score of the selected configuration is optimistic (it won a search), so only the test partition gives an unbiased estimate. No class weights are used, because calibration is fitted explicitly. There is no imputer: the data contract rejects missing values.

## Calibration and threshold (calibration partition only)

The classifier's raw probabilities are scored on the calibration partition. Three methods are compared by **cross-fitted** Brier score inside that partition: none, sigmoid (Platt), isotonic. The lowest Brier wins; ties go to the simpler method. The chosen calibrator is then fitted on all calibration rows. The **decision threshold** maximizes F1 on the cross-fitted calibrated scores of the calibration partition (ties to the higher threshold), so it is not tuned on the fitted calibrator's own output or on test data. With ~120 calibration customers and ~16 exits, both choices are noisy; the report states the counts.

Reliability on the test partition uses 10 equal-width bins with Wilson 95% intervals and flags bins under 20 customers as sparse. Reported calibration measures: Brier score and skill versus the base-rate predictor, expected calibration error, mean predicted minus observed prevalence, and calibration slope.

## Held-out metrics (test partition, evaluated once)

ROC-AUC, PR-AUC (with the prevalence no-skill reference), precision / recall / F1 / confusion matrix at the calibration-chosen threshold, calibration measures, and **recall (and precision, lift) in the top 10% of customers by score**. The 10% review capacity is an illustrative setting (`--review-capacity`), not derived from any bank's workload; ties are broken by raw model score. Sample counts and prevalence accompany every table. Intervals are percentile bootstrap (1000 row resamples, fixed seed); they ignore uncertainty from the split, the selection, and calibration, so they are optimistic.

## Subgroups

Geography, Gender, and Age band, on test customers at the fixed threshold. A group reports metrics only with at least 50 customers and at least 10 exits and 10 non-exits; otherwise only counts, prevalence, and share flagged are shown with an "insufficient data" reason. Slices are descriptive. Differences are not a fairness conclusion, are not corrected for multiple comparison, and say nothing about cause. With the 600-row synthetic demo almost every slice is insufficient; that is the intended behavior.

## Sensitivity analyses

Same estimator and hyperparameters, refit on train, recalibrated on calibration, scored on test:

| Variant | Excludes | Why |
| --- | --- | --- |
| `full` (primary) | none | the model that is saved and scored |
| `no_demographics` | Gender, Age, Geography | sensitivity to demographic attributes |
| `no_uncertain_availability` | Tenure, Balance, NumOfProducts, IsActiveMember, HasCrCard | features the [leakage review](banking-leakage-review.md) cannot show were available before the exit |

Each variant gets ROC-AUC with an interval and a paired-bootstrap difference from the primary. Variants are reported and never change the primary model. They are associations in one snapshot; a small difference does not clear a feature of leakage and a large one does not prove it is leaky. Only the primary is saved as an artifact.

## Lineage and reproducibility

The run ID is `<definition>-<dataset hash[:12]>-<config hash[:8]>`; the config hash covers seed, fractions, search space, feature sets, capacity, and bootstrap count. The same CSV and configuration give the same partitions, selection, metrics, and evaluation file (tested). The report links dataset hash, source kind, contract version, quality-report hash, split hashes, artifact hash, and evaluation-file hash; `register` verifies each before inserting anything, in one transaction.

## Scoring semantics

| Concept | Where | Meaning |
| --- | --- | --- |
| Training / calibration scores | `predictions`, partition from `split_assignments` | in-sample; never performance |
| Held-out test scores | same | out-of-sample, retrospective |
| Evaluation predictions | `evaluation_predictions` | produced during training; source of reported metrics |
| `retrospective_batch`, `retrospective_on_demand` | `predictions.score_kind` | scores for already-labeled snapshot customers |
| Future operational scores | none | not available: no dated data, cutoff, or outcome window |
| NULL lineage / `legacy_unknown` | migrated rows | unverified; no split, no evaluation, not loadable |

## Requirements for genuinely prospective banking prediction

1. Dated observations: an observation (snapshot) date per customer-period and a timestamped event for account closure or whatever "exit" is defined to mean (customer vs account level, dormancy vs closure).
2. A feature cutoff at or before the snapshot date, with each feature's as-of time documented; features recorded after the cutoff (the section 3 candidates) excluded or rebuilt point-in-time.
3. A defined horizon and eligibility rule (for example, active customers with at least N months of history, exit within the next H months), with labels built only from events after the cutoff and censoring handled.
4. Temporal evaluation: train on earlier snapshots, calibrate on a later one, test on the latest; multiple snapshots per customer grouped so one customer is never on both sides.
5. Calibration and threshold re-checked on the most recent period, drift monitoring, and subgroup review on adequate samples, with fairness and permitted-use sign-off for sensitive attributes.
6. A real intervention log and a randomized or well-identified holdout before any claim about retention impact.
7. Per-user authentication, audit logging, and managed migrations before operational scoring is exposed.
